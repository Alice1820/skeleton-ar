#
#
# Copyright (C) 2020 Arrow Asia Pacific Ltd.
#
#

import numpy as np
import cv2
from ctypes import *
import datetime

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
		    ("frameType", c_uint32),
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
lib=ll("./libpicozense_api.so")
# lib=ll("./libvzense_api.so")

# print (lib)

status = c_int()
deviceCount = c_int()
deviceIndex = c_int(0)

status = lib.PsInitialize()
print("initialization status ",status)

# deviceHandle = None
# status = lib.PsOpenDevice(deviceIndex)
# print("open device handle status ",status)
# print (deviceHandle)

status = lib.PsGetDeviceCount(byref(deviceCount))
print("no of device ",deviceCount)

PsNearRange=c_int(0)
PsMiddleRange=c_int(2)
PsFarRange=c_int(5)
# status = lib.PsSetDepthRange(deviceIndex, PsNearRange)
status = lib.PsSetDepthRange(deviceIndex, PsMiddleRange)
# status = lib.PsSetDepthRange(deviceIndex, PsFarRange)
print("Set Depth Range ",status)

status = lib.PsOpenDevice(deviceIndex)
print("open device status ",status)

PsPixelFormatBGR888 = c_int(4)
status = lib.PsSetColorPixelFormat(deviceIndex, PsPixelFormatBGR888)
print("PixelFormat ",PsPixelFormatBGR888)

datamode=c_int(0)
#PsDepthAndRGB_30
status = lib.PsSetDataMode(deviceIndex, datamode)
print("DataMode ",status)

PsDepthSensor = c_int(1)
PsRgbSensor = c_int(2)
cameraParameters = PsCameraParameters()
#cameraParameters.fx = 123.45
#cameraParameters.fy = 2345.554
status = lib.PsGetCameraParameters(deviceIndex, PsDepthSensor, byref(cameraParameters))
print("Fx: {:.3f}".format(cameraParameters.fx))
print("Fy: {:.3f}".format(cameraParameters.fy))

status = lib.PsGetCameraParameters(deviceIndex, PsRgbSensor, byref(cameraParameters))
print("Fx: {:.3f}".format(cameraParameters.fx))
print("Fy: {:.3f}".format(cameraParameters.fy))

cameraExtrinsicParameters = PsCameraExtrinsicParameters()
#cameraExtrinsicParameters.rotation=(1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9)
status = lib.PsGetCameraExtrinsicParameters(deviceIndex, byref(cameraExtrinsicParameters))
#print("Rotation0 : {:.6f}".format(cameraExtrinsicParameters.rotation[0])
print("Rotation:", end = ' ')
for i in cameraExtrinsicParameters.rotation:
	print("{:.6f}".format(i), end = ' ')
print("")

resolution = c_int()
status = lib.PsGetResolution(deviceIndex, byref(resolution))
print("resolution ", resolution)
print("status ", status)

PsDepthFrame = c_int(0)
PsIRFrame = c_int(1)
PsGrayFrame = c_int(2)
PsRGBFrame = c_int(3)
PsMappedRGBFrame = c_int(4)
#FrameType = PsRGBFrame
FrameType = PsDepthFrame
face_detection = True
#print("size ", sizeof(depthFrame))
status = lib.PsStartFrame(deviceIndex, FrameType)
print("Start Frame ",status)
depthFrame = PsFrame()
status = lib.PsReadNextFrame(deviceIndex)
status = lib.PsGetFrame(deviceIndex, FrameType, byref(depthFrame))
frameAddress=depthFrame.pFrameData
length=int(depthFrame.dataLen)
len_i=int(length/2)
pWidth=depthFrame.width
pHeight=depthFrame.height
while True:

	status = lib.PsReadNextFrame(deviceIndex)
	# print("Next Frame ",status)

	status = lib.PsGetFrame(deviceIndex, FrameType, byref(depthFrame))
#for i in range(37):
#	print("{:02X}".format(depthFrame.framedata[i]))

#print("frame Index ", depthFrame.frameIndex)
#print("frame type ", depthFrame.frameType)
#print("pixel format ", depthFrame.pixelFormat)
#print("imuframeno ", depthFrame.imuFrameNo)
#print("Frame data ", depthFrame.pFrameData)
	#print("Frame Len ", depthFrame.dataLen)
#print("width ", depthFrame.width)
#print("height ", depthFrame.height)
#print("GetFrame ",status)

#opencv
#nparr = np.zeros((360, 640, 1), dtype=np.uint8)
	# if (status == 0):
	if (FrameType==PsDepthFrame):
		# frameAddress = depthFrame.pFrameData
		img_data = (len_i * c_uint16).from_address(frameAddress)
		nparr16 = np.array(img_data)
		img16=nparr16.reshape(pHeight, pWidth, 1)
		#print(img16.shape)
		pointxy = (240,320)
		#print(img16[pointxy])
		val=img16[pointxy]
		nparr = nparr16/10
		nparr = np.uint8(nparr)
		#print(nparr)
		img = nparr.reshape(pHeight, pWidth, 1)
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
		print (datetime.datetime.now().__str__())

	# if (FrameType==PsRGBFrame):
	# 	#for rgbframe
	# 	#frameAddress=depthFrame.pFrameData
	# 	img8_data = (length * c_uint8).from_address(frameAddress)
	# 	img8_address = addressof(img8_data)
	# 	nparr8 = np.array(img8_data)
	# 	#print(nparr8)
	# 	img = nparr8.reshape(pHeight, pWidth, 3)

	# 	# if (face_detection):
	# 	# 	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# 	# 	#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	# 	# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 	# 	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# 	# 	for (x,y,w,h) in faces:
	# 	# 		#print("detected")
	# 	# 		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	# 	# 		roi_gray = gray[y:y+h, x:x+w]
	# 	# 		roi_color = img[y:y+h, x:x+w]
	# 	# 		#eyes = eye_cascade.detectMultiScale(roi_gray)
	# 	# 		#for (ex,ey,ew,eh) in eyes:
	# 	# 			#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	# 	# 		face_img = img[y:y+h, x:x+w].copy()
	# 	# 		blob = cv2.dnn.blobFromImage(face_img, 1, (227,227), MODEL_MEAN_VALUES, swapRB=False)
	# 	# 		#Predict Gender
	# 	# 		gender_net.setInput(blob)
	# 	# 		gender_preds = gender_net.forward()
	# 	# 		gender = gender_list[gender_preds[0].argmax()]
	# 	# 		#print("Gender : " + gender)
	# 	# 		#Predict Age
	# 	# 		age_net.setInput(blob)
	# 	# 		age_preds = age_net.forward()
	# 	# 		age = age_list[age_preds[0].argmax()]
	# 	# 		#print("Age Range: " + age)
	# 	# 		overlay_text = "%s %s" % (gender, age)
	# 	# 		cv2.putText(img, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

	# #from irframe
	# if (FrameType==PsIRFrame):
	# 	img16_data = (len * c_uint16).from_address(frameAddress)
	# 	nparr = np.array(img16_data)
	# 	#print("shape ",nparr.shape)
	# 	#print(nparr)
	# 	nparr=nparr/10
	# 	nparr=np.uint8(nparr)
	# 	#print(nparr)
	# 	img = nparr.reshape(pHeight,pWidth,1)

	#print(img.shape)
	img=cv2.imshow('rgb',img)
	key=cv2.waitKey(2)
	if key == 27:
		break
cv2.destroyAllWindows()

#status = lib.PsReadNextFrame(deviceIndex)
#status = lib.PsGetFrame(deviceIndex, FrameType, byref(depthFrame))
#print("frame Index ", depthFrame.frameIndex)
#print("frame Len ", depthFrame.dataLen)
#print("status ", status)

status = lib.PsStopFrame(deviceIndex, FrameType)

status = lib.PsCloseDevice(deviceIndex)
status = lib.PsShutdown()
