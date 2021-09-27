
import sys
# sys.path.insert(1, './pyKinectAzure/')
sys.path.insert(1, './pyKinectAzure/')

import datetime

import numpy as np
import queue
import threading

# from . import pyKinectAzure
from pyKinectAzure import _k4a
from pyKinectAzure.pyKinectAzure import pyKinectAzure
from pyKinectAzure.kinectBodyTracker import kinectBodyTracker, _k4abt
import cv2

import torch
import torchvision.transforms as transforms


class k4aBodyTracker():

	def __init__(self):
		self.max_channel = 3
		self.winLen = 24
		self.vid_len = 8
		self.max_frame = self.winLen
		self.max_joint = 25
		self.max_person = 2
		self.channels = 3
		self.select_person_num = 2
		self.W = 310
		self.H = 256
		# self.dataset = args.dataset
		# self.progress_bar = not args.no_progress_bar
		# self.transform = transform

		# Path to the module
		# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
		# modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
		# bodyTrackingModulePath = 'C:\\Program Files\\Azure Kinect Body Tracking SDK\\sdk\\windows-desktop\\amd64\\release\\bin\\k4abt.dll'
		modulePath = '/usr/lib/x86_64-linux-gnu/libk4a.so'
		bodyTrackingModulePath = '/usr/lib/libk4abt.so'
		# under x86_64 linux please use r'/usr/lib/x86_64-linux-gnu/libk4a.so'
		# In Jetson please use r'/usr/lib/aarch64-linux-gnu/libk4a.so'

		# Initialize the library with the path containing the module
		self.pyK4A = pyKinectAzure(modulePath)

		# Open device
		self.pyK4A.device_open()

		# Modify camera configuration
		device_config = self.pyK4A.config
		device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_OFF
		device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
		# device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_2X2BINNED
		print(device_config)

		# Start cameras using modified configuration
		self.pyK4A.device_start_cameras(device_config)

		# Initialize the body tracker
		self.pyK4A.bodyTracker_start(bodyTrackingModulePath)
		
		# display
		self.imageNow = None
		self._skeNow = [np.zeros((self.max_person, self.max_joint, self.channels)) for _ in range(self.max_frame)]
		self.skeletonNow = np.zeros((self.max_person, self.max_joint, self.channels))
		self._clipNow = [np.zeros((self.H, self.W)) for _ in range(self.max_frame)]
		self.skeNow = self.clipNow = None
		# turn-on the worker thread
		threading.Thread(target=self.next_frame, daemon=True).start()

	# def next_clip(self):
	# 	skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
	# 	with open(file_path, 'r') as fr:
	# 		frame_num = int(fr.readline())
	# 		for frame in range(frame_num):
	# 			person_num = int(fr.readline())
	# 			for person in range(person_num):
	# 				person_info = fr.readline().strip().split()
	# 				joint_num = int(fr.readline())
	# 				for joint in range(joint_num):
	# 					joint_info = fr.readline().strip().split()
	# 					skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32)
	# 	return skeleton[:,:frame_num,:,:], frame_num

	# def load_depth(path, args, vid_len=24):
	# 	img_list = os.listdir(path)
	# 	num_frames = len(img_list)
	# 	width = 224
	# 	height = 224
	# 	# slope = 1600.0
	# 	# dim = (width, height)
	# 	# Init the numpy array
	# 	taken = np.linspace(0, num_frames, vid_len).astype(int)

	# 	np_idx = 0
	# 	for fr_idx in range(num_frames):
	# 		if fr_idx in taken: # 24 frames
	# 			img_path = os.path.join(path, img_list[fr_idx])
	# 			img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) # 16bit
	# 			if not img is None: # skip empty frame
	# 				img = cv2.resize(img, (224, 224)) # 310*256
	# 				# video[np_idx, :, :] = np.array(img, dtype=np.float32) / slope
	# 				video[np_idx, :, :] = np.array(img, dtype=np.float32)
	# 			np_idx += 1
	# 	# print (video[0])
	# 	return video

	def next_skeleton(self):
		self.inputs = 'JVB'
		self.T = self.winLen
		self.conn = connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
		# M, max_frame, V, C -> C, max_frame, V, M
		clip = self.skeNow.transpose(3, 1, 2, 0) # channels, max_frame, 25, max_person
		# (C, max_frame, V, M) -> (I, C*2, T, V, M)
		joint, velocity, bone = self.multi_input(clip[:,:self.T,:,:])
		data_new = []
		if 'J' in self.inputs:
			data_new.append(joint)
		if 'V' in self.inputs:
			data_new.append(velocity)
		if 'B' in self.inputs:
			data_new.append(bone)
		data_new = np.stack(data_new, axis=0)

		return data_new
	
	def depth_transform(self, np_clip):
		####### depth ######
		p_min = 500.
		p_max = 4500.
		np_clip[(np_clip < p_min)] = 0.0
		np_clip[(np_clip > p_max)] = 0.0
		np_clip -= 2500.
		np_clip /= 2000.
		# repeat to BGR to fit pretrained resnet parameters
		np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3

		return np_clip

	def next_clip(self):
		if self.clipNow is not None:
			video = self.depth_transform(self.clipNow)
			# print (video.shape) # (bs, 224, 224, 3)
			video = video.transpose(0, 3, 1, 2) # (bs, 3, 224, 224)
			video = self.NormalizeLen(video, self.vid_len)
			video = self.NumToTensor(video)
			video = self.ToPILImage(video)
			video = self.SpaCenterCrop(video)
			video = self.ToTensor(video)
			return video
		else:
			return None

	def NormalizeLen(self, vid, vid_len=8):
		if vid.shape[0] != 1:
			num_frames = len(vid)
			indices = np.linspace(0, num_frames - 1, vid_len).astype(int)
			vid = vid[indices, :, :]
		return vid

	def NumToTensor(self, vid):
		# return torch.from_numpy(vid.astype(np.float32)).unsqueeze(1).transpose(1, 4).squeeze()
		return torch.from_numpy(vid.astype(np.float32))

	def SpaCenterCrop(self, vid, vid_dim=(224, 224)):
		return [transforms.CenterCrop(vid_dim)(x) for x in vid]

	def ToPILImage(self, vid):
		return [transforms.ToPILImage()(x) for x in vid]

	def ToTensor(self, vid):
		return torch.stack([transforms.ToTensor()(x) for x in vid])

	def multi_input(self, data):
		C, T, V, M = data.shape
		joint = np.zeros((C*2, T, V, M))
		velocity = np.zeros((C*2, T, V, M))
		bone = np.zeros((C*2, T, V, M))
		joint[:C,:,:,:] = data
		for i in range(V):
			joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
		for i in range(T-2):
			velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
			velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
		for i in range(len(self.conn)):
			bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
		bone_length = 0
		for i in range(C):
			bone_length += bone[i,:,:,:] ** 2
		bone_length = np.sqrt(bone_length) + 0.0001
		for i in range(C):
			bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
		return joint, velocity, bone

	def next_frame(self):
		'''
		return skeletons: [max_person, 25, 3]
		'''
		k = 0
		while True:
			# Get capture
			self.pyK4A.device_get_capture()

			# Get the depth image from the capture
			depth_image_handle = self.pyK4A.capture_get_depth_image()

			# Check the image has been read correctly
			if depth_image_handle:

				# Perform body detection
				self.pyK4A.bodyTracker_update()

				# Read and convert the image data to numpy array:
				depth_image = self.pyK4A.image_convert_to_numpy(depth_image_handle)
				depth_color_image = cv2.convertScaleAbs(depth_image, alpha=0.05)  #alpha is fitted by visual comparison with Azure k4aviewer results 
				depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_GRAY2RGB) 

				# Get body segmentation image
				body_image_color = self.pyK4A.bodyTracker_get_body_segmentation()

				combined_image = cv2.addWeighted(depth_color_image, 0.8, body_image_color, 0.2, 0)
				
				# update single image and depth
				self.imageNow = combined_image
				self.depthNow = depth_image

				# Draw the skeleton
				bodies = self.pyK4A.body_tracker.bodiesNow
				for idx, body in enumerate(bodies):
					skeleton2D = self.pyK4A.bodyTracker_project_skeleton(body.skeleton)
					combined_image = self.pyK4A.body_tracker.draw2DSkeleton(skeleton2D, body.id, combined_image)
					# self.pyK4A.body_tracker.printBodyPosition(body)
					# print (skeleton2D.shape)
					# put skeleton in self.skeletonNow
					# print (self.pyK4A.body_tracker.get3DSkeleton(body))
					# print (idx, body.id)
					if idx >= self.max_person:
						break
					self.skeletonNow[idx, :, :] = self.pyK4A.body_tracker.get3DSkeleton(body) # (25, 3)

				# update self.skeNow
				# if len(bodies) > 0:
				for i in range(self.max_frame - 1):
					self._skeNow[i] = np.copy(self._skeNow[i+1])
					self._clipNow[i] = np.copy(self._clipNow[i+1])
				self._skeNow[-1] = np.copy(self.skeletonNow)
				self._clipNow[-1] = np.array(cv2.resize(depth_image, (self.W, self.H))).astype(np.float32) # 224 * 224
				self.skeNow = np.stack(self._skeNow, 0).transpose(1, 0, 2, 3) # [4, 64, :, :]
				self.clipNow = np.stack(self._clipNow, 0) # [64, 224, 224]
				# else:
					# print (datetime.datetime.now().__str__() + '                           ', 'No skeleton detected')
				
				# update self.clipNow
				# print (self.skeNow[0, 0, :, :])
				# # Overlay body segmentation on depth image
				# cv2.imshow('Segmented Depth Image', combined_image)
				# k = cv2.waitKey(1)

				# Release the image
				self.pyK4A.image_release(depth_image_handle)
				self.pyK4A.image_release(self.pyK4A.body_tracker.segmented_body_img)


			self.pyK4A.capture_release()
			self.pyK4A.body_tracker.release_frame()
			# 
			# print (k)

			# if k == 27:
			# 	break
			# elif k == ord('q'):
			# 	cv2.imwrite('outputImage.jpg',combined_image)

			# update self.skeNow
			for j in range(self.max_frame):
				self.skeNow[:, j, :, :] = self.skeletonNow.copy()
		pyK4A.device_stop_cameras()
		pyK4A.device_close()

	def end(self):
		self.pyK4A.device_stop_cameras()
		self.pyK4A.device_close()

k4a2v2 = {
		0: _k4abt.K4ABT_JOINT_PELVIS,
		1: _k4abt.K4ABT_JOINT_SPINE_NAVEL,
		2: _k4abt.K4ABT_JOINT_NECK,
		3: _k4abt.K4ABT_JOINT_HEAD,
		4: _k4abt.K4ABT_JOINT_SHOULDER_LEFT,
		5: _k4abt.K4ABT_JOINT_ELBOW_LEFT,
		6: _k4abt.K4ABT_JOINT_WRIST_LEFT,
		7: _k4abt.K4ABT_JOINT_HAND_LEFT,
		8: _k4abt.K4ABT_JOINT_SHOULDER_RIGHT,
		9: _k4abt.K4ABT_JOINT_ELBOW_RIGHT,
		10: _k4abt.K4ABT_JOINT_WRIST_RIGHT,
		11: _k4abt.K4ABT_JOINT_HAND_RIGHT,
		12: _k4abt.K4ABT_JOINT_HIP_LEFT,
		13: _k4abt.K4ABT_JOINT_KNEE_LEFT,
		14: _k4abt.K4ABT_JOINT_ANKLE_LEFT,
		15: _k4abt.K4ABT_JOINT_FOOT_LEFT,
		16: _k4abt.K4ABT_JOINT_HIP_RIGHT,
		17: _k4abt.K4ABT_JOINT_KNEE_RIGHT,
		18: _k4abt.K4ABT_JOINT_ANKLE_RIGHT,
		19: _k4abt.K4ABT_JOINT_FOOT_RIGHT,
		20: _k4abt.K4ABT_JOINT_SPINE_CHEST,
		21: _k4abt.K4ABT_JOINT_HANDTIP_LEFT,
		22: _k4abt.K4ABT_JOINT_THUMB_LEFT,
		23: _k4abt.K4ABT_JOINT_HANDTIP_RIGHT,
		24: _k4abt.K4ABT_JOINT_THUMB_RIGHT,
		 }

if __name__ == "__main__":
	
	_k4abt.k4abt.setup_library('/usr/lib/libk4abt.so')
	print (k4a2v2)
	tracker = k4aBodyTracker()
	tracker.next_frame()

