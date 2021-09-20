
import sys
sys.path.insert(1, './pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a
from kinectBodyTracker import kinectBodyTracker, _k4abt
import cv2

class k4aBodyTracking():

	def __init__(self):
		self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform
		self.winLen = 32

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

	def next(self):
		return None

	def next_clip(self):
        skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        with open(file_path, 'r') as fr:
            frame_num = int(fr.readline())
            for frame in range(frame_num):
                person_num = int(fr.readline())
                for person in range(person_num):
                    person_info = fr.readline().strip().split()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        joint_info = fr.readline().strip().split()
                        skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32)
        return skeleton[:,:frame_num,:,:], frame_num

	def next_frame(self):
		k = 0
		# while True:
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
				self.pyK4A.body_tracker.printBodyPosition(body)
			# Overlay body segmentation on depth image
			cv2.imshow('Segmented Depth Image',combined_image)
			k = cv2.waitKey(1)

		# Release the image
		self.pyK4A.image_release(depth_image_handle)
		self.pyK4A.image_release(self.pyK4A.body_tracker.segmented_body_img)


		self.pyK4A.capture_release()
		self.pyK4A.body_tracker.release_frame()

		if k == ord('q'):
			cv2.imwrite('outputImage.jpg',combined_image)

	def end(self):
		self.pyK4A.device_stop_cameras()
		self.pyK4A.device_close()
		
# if __name__ == "__main__":
# 	continue

