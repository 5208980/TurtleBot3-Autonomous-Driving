#!/usr/bin/env python

from __future__ import division
import numpy as np
import math
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist


class ImageTransform():
	def __init__(self):
		self.name = "ImageTransform ::"
		self.bridge = CvBridge()

		# self.sub_lane = rospy.Subscriber('/raspicam_node/image/compressed'', CompressedImage, self.image_cb, queue_size=1)
		self.sub_lane = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_cb, queue_size=1)

		self.pub_bird_eye = rospy.Publisher('/transform/bird_eye', Image, queue_size=1)
		self.pub_binary = rospy.Publisher('/transform/binary', Image, queue_size=1)

	# Default bgr8 encoding
	def convert_compressed_image_to_cv(self, msg):
		frame_arr = np.fromstring(msg.data, np.uint8)
		frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
		return frame

	# image in black or white
	def create_threshold_binary_image(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.blur(frame,(5,5))
		_, frame = cv2.threshold(frame, 130 ,255, cv2.THRESH_BINARY)
		binary = cv2.blur(frame,(5,5))
		return binary

	# perspective transform (bird-eye view)
	def bird_eye_perspective_transform(self, frame):
		top_x = 170
		top_y = 50
		bottom_x = 300
		bottom_y = 170

		pts_src = np.array([
			[320 - top_x, 360 - top_y],
			[320 + top_x, 360 - top_y],
			[320 + bottom_x, 240 + bottom_y],
			[320 - bottom_x, 240 + bottom_y]])
		pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])

		homo, status = cv2.findHomography(pts_src, pts_dst)
		warped = cv2.warpPerspective(frame, homo, (1000, 600))
		return warped

	def image_cb(self, msg):
		frame = self.convert_compressed_image_to_cv(msg)

		frame = self.bird_eye_perspective_transform(frame)
		self.pub_bird_eye.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

		frame = self.create_threshold_binary_image(frame)
		self.pub_binary.publish(self.bridge.cv2_to_imgmsg(frame, "passthrough"))

	def main(self):
		rospy.loginfo("%s Spinning", self.name)
		rospy.spin()

if __name__ == '__main__': # sys.argv
	rospy.init_node('image_transform')
	node = ImageTransform()
	node.main()
