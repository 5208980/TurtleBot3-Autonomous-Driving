#!/usr/bin/env python

import rospy
import numpy as np
import math
import os
import cv2
from enum import Enum
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


class DetectSign():
	def __init__(self):
		self.onload()		# Initalise sift, image

		self.cvBridge = CvBridge()
		self.sub_img_type = "raw" # you can choose image type "compressed", "raw"
		self.pub_image_type = "compressed" # you can choose image type "compressed", "raw"

		if self.sub_img_type == "compressed":
			self.sub_img = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.cbStopSign, queue_size = 1)
		elif self.sub_img_type == "raw":		# when using custom raw
			self.sub_img = rospy.Subscriber('/raspicam_node/image/raw', Image, self.cbStopSign, queue_size = 1)

		self.pub_traffic_sign = rospy.Publisher('/detect/traffic_sign', UInt8, queue_size=1)

		if self.pub_image_type == "compressed":
			# publishes traffic sign image in compressed type 
			self.pub_image_traffic_sign = rospy.Publisher('/detect/image_output/compressed', CompressedImage, queue_size = 1)
		elif self.pub_image_type == "raw":
			# publishes traffic sign image in raw type
			self.pub_image_traffic_sign = rospy.Publisher('/detect/image_output', Image, queue_size = 1)

		self.TrafficSign = Enum('TrafficSign', 'divide stop parking tunnel')
		self.counter = 1

	def onload(self):
		# Initiate SIFT detector
		self.sift = cv2.xfeatures2d.SIFT_create()

		# get the stop sign image in assets
		dir_path = os.path.dirname(os.path.realpath(__file__))
		dir_path = dir_path.replace('project-2/scripts', 'project-2/assets')
		self.stop_sign_img = cv2.imread(dir_path + 'stop.jpg', 0)

		# stop sign image features
		self.kp2, self.des2 = self.sift.detectAndCompute(self.stop_sign_img, None)

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)

		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	# Detects stop sign and then compute the distance to know when to stop
	def cbFindTrafficSign(self, image_msg):
		# Drop frame to 1/5 (6fps) because of the processing speed.
		if self.counter % 3 != 0:
			self.counter += 1
			return
		else:
			self.counter = 1

		# Ros Img -> Cv Img
		if self.sub_image_type == "compressed":
			np_arr = np.fromstring(image_msg.data, np.uint8)
			cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		elif self.sub_image_type == "raw":
			cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, "bgr8")

		MIN_MATCH_COUNT = 9
		MIN_MSE_DECISION = 50000

		kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)	# extract features of camera

		matches2 = self.flann.knnMatch(des1,self.des2,k=2)	# match camera with stop.jpg

		rospy.loginfo(matches2)

	def main(self):
		rospy.spin()

if __name__ == '__main__':
	rospy.init_node('detect_sign')
	node = DetectSign()
	node.main()