import sys
sys.path.insert(1, './self.lib/')
import os
import datetime

import numpy as np
import queue
import threading
import time 

# from . import pyKinectAzure
import cv2

import torch
import torchvision.transforms as transforms
from ctypes import *

PIXEL_MAX = 4500

class PsCameraParameters(Structure):
	_fields_ = [("fx", c_double),
		    ("fy", c_double),
		    ("cx", c_double),
		    ("cy", c_double),
		    ("k1", c_double),
		    ("k2", c_double),
		    ("p1", c_double),
		    ("p2", c_double),
		    ("k3", c_double),
		    ("k4", c_double),
		    ("k5", c_double),
		    ("k6", c_double)]

class PsCameraExtrinsicParameters(Structure):
	_fields_ = [("rotation", c_double * 9), ("translation", c_double * 3)]

class PsFrameMode(Structure):
	_fields_ = [("pixelFormat", c_uint8),
		    ("resolutionWidth", c_int32),
		    ("resulutionHeight", c_int32),
		    ("fps", c_int32)]

class PsFrame(Structure):
	_pack_ = 1
	_fields_ = [("frameIndex", c_uint32),
		    ("self.FrameType", c_uint32),
		    ("pixelFormat", c_uint32),
		    ("imuFrameNo", c_uint8),
		    ("pFrameData", c_void_p),
		    ("dataLen", c_uint32),
		    ("exposuretime", c_float),
		    ("depthRange", c_uint32),
		    ("width", c_uint16),
		    ("height", c_uint16)]

class PsFrameTry(Structure):
	_fields_ = [("framedata", c_uint8 *37)]

class zenseTracker():

	def __init__(self, args):
		self.max_channel = 3
		self.winLen = 32	 # should tune to real-time fps
		# self.winLen = 64 # should tune to real-time fps
		self.vid_len = args.num_segments
		self.max_frame = self.winLen
		self.max_joint = 25
		self.max_person = 2
		self.channels = 3
		self.select_person_num = 2
		self.W = 310
		self.H = 256

		# initialize device
		MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
		age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
		gender_list = ['Male', 'Female']
		#def load_caffe_models():
		#age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
		#gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
		#	return(age_net, gender_net)

		#def video_detector(age_net, gender_net):
		font = cv2.FONT_HERSHEY_SIMPLEX

		ll=cdll.LoadLibrary
		self.lib=ll("./libpicozense_api.so")

		status = c_int()
		self.deviceCount = c_int()
		self.deviceIndex = c_int(0)

		status = self.lib.PsInitialize()
		print("status ",status)

		status = self.lib.PsGetDeviceCount(byref(self.deviceCount))
		print("no of device ",self.deviceCount)

		PsNearRange=c_int(0)
		PsMiddleRange=c_int(1)
		PsFarRange=c_int(2)
		status = self.lib.PsSetDepthRange(self.deviceIndex, PsFarRange)
		print("Set Depth Range ", status)

		status = self.lib.PsOpenDevice(self.deviceIndex)
		print("open device status ", status)

		PsPixelFormatBGR888 = c_int(4)
		status = self.lib.PsSetColorPixelFormat(self.deviceIndex, PsPixelFormatBGR888)
		print("PixelFormat ", PsPixelFormatBGR888)

		datamode=c_int(0)
		#PsDepthAndRGB_30
		status = self.lib.PsSetDataMode(self.deviceIndex, datamode)
		print("DataMode ",status)

		PsDepthSensor = c_int(1)
		PsRgbSensor = c_int(2)
		cameraParameters = PsCameraParameters()
		#cameraParameters.fx = 123.45
		#cameraParameters.fy = 2345.554
		status = self.lib.PsGetCameraParameters(self.deviceIndex, PsDepthSensor, byref(cameraParameters))
		print("Fx: {:.3f}".format(cameraParameters.fx))
		print("Fy: {:.3f}".format(cameraParameters.fy))

		status = self.lib.PsGetCameraParameters(self.deviceIndex, PsRgbSensor, byref(cameraParameters))
		print("Fx: {:.3f}".format(cameraParameters.fx))
		print("Fy: {:.3f}".format(cameraParameters.fy))

		cameraExtrinsicParameters = PsCameraExtrinsicParameters()
		#cameraExtrinsicParameters.rotation=(1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9)
		status = self.lib.PsGetCameraExtrinsicParameters(self.deviceIndex, byref(cameraExtrinsicParameters))
		#print("Rotation0 : {:.6f}".format(cameraExtrinsicParameters.rotation[0])
		print("Rotation:", end = ' ')
		for i in cameraExtrinsicParameters.rotation:
			print("{:.6f}".format(i), end = ' ')
		print("")

		resolution = c_int()
		status = self.lib.PsGetResolution(self.deviceIndex, byref(resolution))
		print("resolution ", resolution)
		print("status ", status)

		self.PsDepthFrame = c_int(0)
		#self.FrameType = PsRGBFrame
		self.FrameType = self.PsDepthFrame
		face_detection = True
		#print("size ", sizeof(self.depthFrame))
		status = self.lib.PsStartFrame(self.deviceIndex, self.FrameType)
		print("Start Frame ",status)
		self.depthFrame = PsFrame()
		status = self.lib.PsReadNextFrame(self.deviceIndex)
		status = self.lib.PsGetFrame(self.deviceIndex, self.FrameType, byref(self.depthFrame))
		self.frameAddress=self.depthFrame.pFrameData
		length = int(self.depthFrame.dataLen)
		self.len_i = int(length / 2)
		self.pWidth=self.depthFrame.width
		self.pHeight=self.depthFrame.height

		# display
		self.imageNow = None
		self.clipNow = None
		self.depthNow = None
		self.cmapImgNow = None
		self.normImgNow = None
		self.skeNow = None
		self.cmapNow = None
		self.normNow = None
		self.ts = []
		self._skeNow = [np.zeros((self.max_person, self.max_joint, self.channels)) for _ in range(self.max_frame)]
		self.skeletonNow = np.zeros((self.max_person, self.max_joint, self.channels))
		self._clipNow = [np.zeros((self.H, self.W)) for _ in range(self.max_frame)]
		self._normNow = [np.zeros((self.H, self.W)) for _ in range(self.max_frame)]
		self._cmapNow = [np.zeros((self.H, self.W, 3)) for _ in range(self.max_frame)]
		# turn-on the worker thread
		threading.Thread(target=self.next_frame, daemon=True).start()

		self.args = args

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
	
	def depthn_transform(self, np_clip):
		# repeat to BGR to fit pretrained resnet parameters
		np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3
		return np_clip

	def depth_transform(self, np_clip):
		####### depth ######
		# p_min = 500.
		p_min = 1500.
		p_max = 4500.
		np_clip[(np_clip < p_min)] = 0.0
		np_clip[(np_clip > p_max)] = 0.0
		np_clip -= 2500.
		np_clip /= 2000.
		# repeat to BGR to fit pretrained resnet parameters
		np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3

		return np_clip

	def next_clip(self):
		if self.args.depthmode == 'depthp' or self.args.depthmode == 'depth':
			if self.clipNow is not None:
				video = self.depth_transform(self.clipNow)
				# print (video.shape) # (bs, 224, 224, 3)
				video = video.transpose(0, 3, 1, 2) # (bs, 3, 224, 224)
				video = self.NormalizeLen(video, self.vid_len)
				video = self.NumToTensor(video)
				# video = self.ToPILImage(video)
				# video = self.SpaCenterCrop(video)
				# video = self.ToTensor(video)
				return video
			else:
				return None
		elif self.args.depthmode == 'depthn' or self.args.depthmode == 'depthnv':
			if self.normNow is not None:
				video = self.depthn_transform(self.normNow)
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
		elif self.args.depthmode == 'depthc':
			if self.cmapNow is not None:
				video = self.video_transform(self.cmapNow)
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
		start_time = time.time()
		while True:
			status = self.lib.PsReadNextFrame(self.deviceIndex)
			status = self.lib.PsGetFrame(self.deviceIndex, self.FrameType, byref(self.depthFrame))
			if (self.FrameType==self.PsDepthFrame):
				#self.frameAddress = self.depthFrame.pFrameData
				img_data=(self.len_i * c_uint16).from_address(self.frameAddress)
				nparr16 = np.array(img_data).astype(np.float32)
				img16=nparr16.reshape(self.pHeight, self.pWidth, 1)
				#print(img16.shape)
				pointxy = (240,320)
				#print(img16[pointxy])
				val=img16[pointxy]
				nparr = nparr16/10
				nparr = np.uint8(nparr)
				#print(nparr)
				img = nparr.reshape(self.pHeight, self.pWidth, 1)
				img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
				pointxy = (320,240)
				radius = 5
				color = (255,0,0)
				thickness = 2
				cv2.circle(img,pointxy,radius, color,thickness)
				font = cv2.FONT_HERSHEY_SIMPLEX
				org = (50,50)
				fontScale = 1
				overlay_text = "%s" % (val)
				cv2.putText(img,overlay_text,pointxy,font,fontScale,color, thickness,cv2.LINE_AA)

				# update single image and depth
				self.imageNow = img
				self.depthNow = img16.astype(np.uint16)
				self.normImgNow = self.depthNow / PIXEL_MAX * 255
				self.normImgNow = np.array(self.normImgNow, dtype=np.uint8)
				# self.cmapImgNow = cv2.applyColorMap(self.normImgNow, cv2.COLORMAP_PARULA)
				self.cmapImgNow = cv2.applyColorMap(self.normImgNow, cv2.COLORMAP_RAINBOW)
				if self.args.depthmode == 'depthv':
					self.normImgNow = self.depthNow / np.max(self.depthNow)
				elif self.args.depthmode == 'depthn':
					pass
				# update self.skeNow
				# if len(bodies) > 0:
				for i in range(self.max_frame - 1):
					self._clipNow[i] = np.copy(self._clipNow[i+1]) # list
					self._normNow[i] = np.copy(self._normNow[i+1])
					self._cmapNow[i] = np.copy(self._cmapNow[i+1])
					if len(self.ts) == self.winLen:
						self.ts[i] = self.ts[i+1]
						self.ts[-1] = datetime.datetime.now().__str__()
					else:
						self.ts.append(datetime.datetime.now().__str__())
				self._clipNow[-1] = np.array(cv2.resize(self.depthNow, (self.W, self.H))).astype(np.float32) # 224 * 224
				self._normNow[-1] = np.array(cv2.resize(self.normImgNow, (self.W, self.H))).astype(np.float32) # 224 * 224
				self._cmapNow[-1] = np.array(cv2.resize(self.cmapImgNow, (self.W, self.H))).astype(np.float32) # 224 * 224
				self.clipNow = np.stack(self._clipNow, 0) # [64, 224, 224]
				self.normNow = np.stack(self._normNow, 0) # [64, 224, 224]
				self.cmapNow = np.stack(self._cmapNow, 0) # [64, 224, 224]
				self.fps = (1 / (time.time() - start_time))
				# should set window length to 1.5s * self.fps
				start_time = time.time()
				# print (self.fps)
				# else:
					# print (datetime.datetime.now().__str__() + '                           ', 'No skeleton detected')
				# print (datetime.datetime.now().__str__())
				# update self.clipNow	
				# print (self.skeNow[0, 0, :, :])
				# # Overlay body segmentation on depth image
				# cv2.imshow('Segmented Depth Image', combined_image)
		cv2.destroyAllWindows()


class zenseLoader():

	def __init__(self, args):
		self.max_channel = 3
		self.winLen = 36 # should tune to real-time fps
		# self.winLen = 64 # should tune to real-time fps
		self.vid_len = args.num_segments
		self.max_frame = self.winLen
		self.max_joint = 25
		self.max_person = 2
		self.channels = 3
		self.select_person_num = 2
		self.W = 310
		self.H = 256

		self.test_dir = './test/depth'
		self.test_list = sorted(os.listdir(self.test_dir))
		self.LEN = len(self.test_list)
		print (self.LEN)
		# # initialize device
		# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
		# age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
		# gender_list = ['Male', 'Female']
		# #def load_caffe_models():
		# #age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
		# #gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
		# #	return(age_net, gender_net)

		# #def video_detector(age_net, gender_net):
		# font = cv2.FONT_HERSHEY_SIMPLEX

		# ll=cdll.LoadLibrary
		# self.lib=ll("./libpicozense_api.so")

		# status = c_int()
		# self.deviceCount = c_int()
		# self.deviceIndex = c_int(0)

		# status = self.lib.PsInitialize()
		# print("status ",status)

		# status = self.lib.PsGetDeviceCount(byref(self.deviceCount))
		# print("no of device ",self.deviceCount)

		# PsNearRange=c_int(0)
		# PsMiddleRange=c_int(1)
		# PsFarRange=c_int(2)
		# status = self.lib.PsSetDepthRange(self.deviceIndex, PsFarRange)
		# print("Set Depth Range ", status)

		# status = self.lib.PsOpenDevice(self.deviceIndex)
		# print("open device status ", status)

		# PsPixelFormatBGR888 = c_int(4)
		# status = self.lib.PsSetColorPixelFormat(self.deviceIndex, PsPixelFormatBGR888)
		# print("PixelFormat ", PsPixelFormatBGR888)

		# datamode=c_int(0)
		# #PsDepthAndRGB_30
		# status = self.lib.PsSetDataMode(self.deviceIndex, datamode)
		# print("DataMode ",status)

		# PsDepthSensor = c_int(1)
		# PsRgbSensor = c_int(2)
		# cameraParameters = PsCameraParameters()
		# #cameraParameters.fx = 123.45
		# #cameraParameters.fy = 2345.554
		# status = self.lib.PsGetCameraParameters(self.deviceIndex, PsDepthSensor, byref(cameraParameters))
		# print("Fx: {:.3f}".format(cameraParameters.fx))
		# print("Fy: {:.3f}".format(cameraParameters.fy))

		# status = self.lib.PsGetCameraParameters(self.deviceIndex, PsRgbSensor, byref(cameraParameters))
		# print("Fx: {:.3f}".format(cameraParameters.fx))
		# print("Fy: {:.3f}".format(cameraParameters.fy))

		# cameraExtrinsicParameters = PsCameraExtrinsicParameters()
		# #cameraExtrinsicParameters.rotation=(1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9)
		# status = self.lib.PsGetCameraExtrinsicParameters(self.deviceIndex, byref(cameraExtrinsicParameters))
		# #print("Rotation0 : {:.6f}".format(cameraExtrinsicParameters.rotation[0])
		# print("Rotation:", end = ' ')
		# for i in cameraExtrinsicParameters.rotation:
		# 	print("{:.6f}".format(i), end = ' ')
		# print("")

		# resolution = c_int()
		# status = self.lib.PsGetResolution(self.deviceIndex, byref(resolution))
		# print("resolution ", resolution)
		# print("status ", status)

		# self.PsDepthFrame = c_int(0)
		# #self.FrameType = PsRGBFrame
		# self.FrameType = self.PsDepthFrame
		# face_detection = True
		# #print("size ", sizeof(self.depthFrame))
		# status = self.lib.PsStartFrame(self.deviceIndex, self.FrameType)
		# print("Start Frame ",status)
		# self.depthFrame = PsFrame()
		# status = self.lib.PsReadNextFrame(self.deviceIndex)
		# status = self.lib.PsGetFrame(self.deviceIndex, self.FrameType, byref(self.depthFrame))
		# self.frameAddress=self.depthFrame.pFrameData
		# length = int(self.depthFrame.dataLen)
		# self.len_i = int(length / 2)
		# self.pWidth=self.depthFrame.width
		# self.pHeight=self.depthFrame.height

		# display
		self.imageNow = None
		self.clipNow = None
		self.depthNow = None
		self.cmapImgNow = None
		self.normImgNow = None
		self.skeNow = None
		self.cmapNow = None
		self.normNow = None
		self.ts = []
		self._skeNow = [np.zeros((self.max_person, self.max_joint, self.channels)) for _ in range(self.max_frame)]
		self.skeletonNow = np.zeros((self.max_person, self.max_joint, self.channels))
		self._clipNow = [np.zeros((self.H, self.W)) for _ in range(self.max_frame)]
		self._normNow = [np.zeros((self.H, self.W)) for _ in range(self.max_frame)]
		self._cmapNow = [np.zeros((self.H, self.W, 3)) for _ in range(self.max_frame)]
		# turn-on the worker thread
		threading.Thread(target=self.next_frame, daemon=True).start()

		self.args = args

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
	
	def depthn_transform(self, np_clip):
		# repeat to BGR to fit pretrained resnet parameters
		np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3
		return np_clip

	def depth_transform(self, np_clip):
		####### depth ######
		# p_min = 500.
		p_min = 1500.
		p_max = 4500.
		np_clip[(np_clip < p_min)] = 0.0
		np_clip[(np_clip > p_max)] = 0.0
		np_clip -= 2500.
		np_clip /= 2000.
		# repeat to BGR to fit pretrained resnet parameters
		np_clip = np.repeat(np_clip[:, :, :, np.newaxis], 3, axis=3) # 24, 310, 256, 3

		return np_clip

	def next_clip(self):
		if self.args.depthmode == 'depthp' or self.args.depthmode == 'depth':
			if self.clipNow is not None:
				video = self.depth_transform(self.clipNow)
				# print (video.shape) # (bs, 224, 224, 3)
				video = video.transpose(0, 3, 1, 2) # (bs, 3, 224, 224)
				video = self.NormalizeLen(video, self.vid_len)
				video = self.NumToTensor(video)
				# video = self.ToPILImage(video)
				# video = self.SpaCenterCrop(video)
				# video = self.ToTensor(video)
				return video
			else:
				return None
		elif self.args.depthmode == 'depthn' or self.args.depthmode == 'depthnv':
			if self.normNow is not None:
				video = self.depthn_transform(self.normNow)
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
		elif self.args.depthmode == 'depthc':
			if self.cmapNow is not None:
				video = self.video_transform(self.cmapNow)
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
		j = 0
		while True:
			j += 1
			idx = j % self.LEN
			img16 = cv2.imread(os.path.join(self.test_dir, self.test_list[idx]), -1)
			# print (idx,self.test_list[idx])
			img = img16.astype(np.float32)
			img = np.uint8(img / 10)
			img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
			# status = self.lib.PsReadNextFrame(self.deviceIndex)
			# status = self.lib.PsGetFrame(self.deviceIndex, self.FrameType, byref(self.depthFrame))
			# if (self.FrameType==self.PsDepthFrame):
			# 	#self.frameAddress = self.depthFrame.pFrameData
			# 	img_data=(self.len_i * c_uint16).from_address(self.frameAddress)
			# 	nparr16 = np.array(img_data).astype(np.float32)
			# 	img16=nparr16.reshape(self.pHeight, self.pWidth, 1)
			# 	#print(img16.shape)
			# 	pointxy = (240,320)
			# 	#print(img16[pointxy])
			# 	val=img16[pointxy]
			# 	nparr = nparr16/10
			# 	nparr = np.uint8(nparr)
			# 	#print(nparr)
			# 	img = nparr.reshape(self.pHeight, self.pWidth, 1)
			# 	img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
			# 	pointxy = (320,240)
			# 	radius = 5
			# 	color = (255,0,0)
			# 	thickness = 2
			# 	cv2.circle(img,pointxy,radius, color,thickness)
			# 	font = cv2.FONT_HERSHEY_SIMPLEX
			# 	org = (50,50)
			# 	fontScale = 1
			# 	overlay_text = "%s" % (val)
			# 	cv2.putText(img,overlay_text,pointxy,font,fontScale,color, thickness,cv2.LINE_AA)

			# update single image and depth
			self.imageNow = img
			self.depthNow = img16.astype(np.uint16)
			self.normImgNow = self.depthNow / PIXEL_MAX * 255
			self.normImgNow = np.array(self.normImgNow, dtype=np.uint8)
			# self.cmapImgNow = cv2.applyColorMap(self.normImgNow, cv2.COLORMAP_PARULA)
			self.cmapImgNow = cv2.applyColorMap(self.normImgNow, cv2.COLORMAP_RAINBOW)
			if self.args.depthmode == 'depthv':
				self.normImgNow = self.depthNow / np.max(self.depthNow)
			elif self.args.depthmode == 'depthn':
				pass
			# update self.skeNow
			# if len(bodies) > 0:
			for i in range(self.max_frame - 1):
				self._clipNow[i] = np.copy(self._clipNow[i+1]) # list
				self._normNow[i] = np.copy(self._normNow[i+1])
				self._cmapNow[i] = np.copy(self._cmapNow[i+1])
				if len(self.ts) == self.winLen:
					self.ts[i] = self.ts[i+1]
					self.ts[-1] = datetime.datetime.now()
				else:
					self.ts.append(datetime.datetime.now())
			self._clipNow[-1] = np.array(cv2.resize(self.depthNow, (self.W, self.H))).astype(np.float32) # 224 * 224
			self._normNow[-1] = np.array(cv2.resize(self.normImgNow, (self.W, self.H))).astype(np.float32) # 224 * 224
			self._cmapNow[-1] = np.array(cv2.resize(self.cmapImgNow, (self.W, self.H))).astype(np.float32) # 224 * 224
			self.clipNow = np.stack(self._clipNow, 0) # [64, 224, 224]
			self.normNow = np.stack(self._normNow, 0) # [64, 224, 224]
			self.cmapNow = np.stack(self._cmapNow, 0) # [64, 224, 224]
			# print (self.ts[-1] - self.ts[0])
			# else:
				# print (datetime.datetime.now().__str__() + '                           ', 'No skeleton detected')
			# print (datetime.datetime.now().__str__())
			# update self.clipNow	
			# print (self.skeNow[0, 0, :, :])
			# # Overlay body segmentation on depth image
			# cv2.imshow('Segmented Depth Image', combined_image)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	
	_k4abt.k4abt.setup_library('/usr/lib/libk4abt.so')
	print (k4a2v2)
	tracker = k4aBodyTracker()
	tracker.next_frame()

