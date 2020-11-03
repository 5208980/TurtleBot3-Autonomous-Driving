#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

class GroundViewNode():
	def __init__(self):
		self.clip_hist_percent = 1.
		self.name = 'GroundViewNode ::'
		self.bridge = CvBridge()
		self.sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.ground_view_cb)
		self.pub = rospy.Publisher('/raspicam_node/image/ground_view', CompressedImage, queue_size=1)
		self.top_x = 120
		self.top_y = 12
		self.bottom_x = 316
		self.bottom_y = 120

	def ground_view_cb(self, frame):
		# converts compressed image to opencv image
		# np_image_original = np.fromstring(frame.data, np.uint8)
		# cv_image_original = cv2.imdecode(np_image_original, cv2.IMREAD_COLOR)
		np_image_original = np.fromstring(frame.data, np.uint8)
		cv_image_original = cv2.imdecode(np_image_original, cv2.IMREAD_COLOR)

		# cv_image_compensated = np.copy(cv_image_original)

		cv_image_original = cv2.GaussianBlur(cv_image_original, (5, 5), 0)
		# cv_image_original = cv2.cvtColor(cv_image_original, cv2.COLOR_BGR2RGB)
		## homography transform process
		# selecting 4 points from the original image
		top_x = self.top_x
		top_y = self.top_y
		bottom_x = self.bottom_x
		bottom_y = self.bottom_y

		pts_src = np.array([[160 - 120, 180 - 12], [160 + 120, 180 - 12], [160 + 316, 120 + 120], [160 - 316, 120 + 120]])

		# selecting 4 points from image that will be transformed
		pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])

		# finding homography matrix
		h, status = cv2.findHomography(pts_src, pts_dst)

		# homography process
		cv_image_homography = cv2.warpPerspective(cv_image_original, h, (1000, 600))

		# fill the empty space with black triangles on left and right side of bottom
		triangle1 = np.array([[0, 599], [0, 340], [200, 599]], np.int32)
		triangle2 = np.array([[999, 599], [999, 340], [799, 599]], np.int32)
		black = (0, 0, 0)
		white = (255, 255, 255)
		cv_image_homography = cv2.fillPoly(cv_image_homography, [triangle1, triangle2], black)

		img_msg = self.bridge.cv2_to_compressed_imgmsg(cv_image_homography, "jpg")
		self.pub.publish(img_msg)

	def main(self):
		rospy.loginfo("%s Spinning", self.name)
		rospy.spin()

if __name__ == '__main__': # sys.argv
	rospy.init_node('ground_view')
	node = GroundViewNode()
	node.main()
