
import sys
# sys.path.insert(1, './pyKinectAzure/')
sys.path.insert(1, './src/pyKinectAzure/')

import numpy as np
import queue
import threading

# from . import pyKinectAzure
from .pyKinectAzure import _k4a
from .pyKinectAzure.pyKinectAzure import pyKinectAzure
from .pyKinectAzure.kinectBodyTracker import kinectBodyTracker, _k4abt
import cv2

class k4aBodyTracker():

	def __init__(self):
		self.max_channel = 3
		self.max_frame = 300
		self.max_joint = 25
		self.max_person = 4
		self.channels = 3
		self.select_person_num = 2
		# self.dataset = args.dataset
		# self.progress_bar = not args.no_progress_bar
		# self.transform = transform
		self.winLen = 96

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
		print(device_config)

		# Start cameras using modified configuration
		self.pyK4A.device_start_cameras(device_config)

		# Initialize the body tracker
		self.pyK4A.bodyTracker_start(bodyTrackingModulePath)
		
		# display
		self.imageNow = None
		self.clipNow = np.zeros((self.max_person, self.max_frame, self.max_joint, self.channels))
		self.skeletonNow = np.zeros((self.max_person, self.max_joint, self.channels))
		# turn-on the worker thread
		threading.Thread(target=self.next_frame, daemon=True).start()

	def next(self):
		return None

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

	def next_clip(self):
		self.inputs = 'JVB'
		self.T = 96
		self.conn = connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
		clip = self.clipNow.transpose(3, 1, 2, 0) # channels, max_frame, 25, max_person
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
				depth_color_image = cv2.convertScaleAbs (depth_image, alpha=0.05)  #alpha is fitted by visual comparison with Azure k4aviewer results 
				depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_GRAY2RGB) 

				# Get body segmentation image
				body_image_color = self.pyK4A.bodyTracker_get_body_segmentation()

				combined_image = cv2.addWeighted(depth_color_image, 0.8, body_image_color, 0.2, 0)

				# Draw the skeleton
				for body in self.pyK4A.body_tracker.bodiesNow:
					skeleton2D = self.pyK4A.bodyTracker_project_skeleton(body.skeleton)
					combined_image = self.pyK4A.body_tracker.draw2DSkeleton(skeleton2D, body.id, combined_image)
					# self.pyK4A.body_tracker.printBodyPosition(body)
					# print (skeleton2D.shape)
					# put skeleton in self.skeletonNow
					# print (self.pyK4A.body_tracker.get3DSkeleton(body))
					self.skeletonNow[body.id - 1, :, :] = self.pyK4A.body_tracker.get3DSkeleton(body) # (25, 3)
					if body.id > self.max_person:
						break

				for i in range(self.max_frame - 1):
					self.clipNow[:, i, :, :] = self.clipNow[:, i+1, :, :]
				self.clipNow[:, -1, :, :] = self.skeletonNow
				self.imageNow = combined_image
				# print (self.clipNow[0, 0, :, :])
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

			# update self.clipNow
			for j in range(self.max_frame):
				self.clipNow[:, j, :, :] = self.skeletonNow.copy()
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

