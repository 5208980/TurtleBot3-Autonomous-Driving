#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

class ImageCompensation():
	def __init__(self):
		self.clip_hist_percent = 1.
		self.bridge = CvBridge()
		self.sub_image_original = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.cbImageCompensation, queue_size = 1)
		self.pub_image_compensated = rospy.Publisher('/raspicam_node/image/image_output', CompressedImage, queue_size = 1)

	def cbImageCompensation(self, frame):
		frame_arr = np.fromstring(frame.data, np.uint8)
		frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
		# cv_image_compensated = np.copy(cv_image_original)

		## Image compensation based on pseudo histogram equalization
		clip_hist_percent = self.clip_hist_percent

		hist_size = 256
		min_gray = 0
		max_gray = 0
		alpha = 0
		beta = 0

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# histogram calculation
		if clip_hist_percent == 0.0:
			min_gray, max_gray, _, _ = cv2.minMaxLoc(frame_gray)
		else:
			hist = cv2.calcHist([frame_gray], [0], None, [hist_size], [0, hist_size])

		accumulator = np.cumsum(hist)

		max = accumulator[hist_size - 1]

		clip_hist_percent *= (max / 100.)
		clip_hist_percent /= 2.

		min_gray = 0
		while accumulator[min_gray] < clip_hist_percent:
			min_gray += 1

		max_gray = hist_size - 1
		while accumulator[max_gray] >= (max - clip_hist_percent):
			max_gray -= 1

		input_range = max_gray - min_gray

		alpha = (hist_size - 1) / input_range
		beta = -min_gray * alpha

		frame_gray = cv2.convertScaleAbs(frame_gray, -1, alpha, beta)

		img_msg = self.bridge.cv2_to_compressed_imgmsg(frame_gray, "jpg")
		self.pub_image_compensated.publish(img_msg)

	def main(self):
		rospy.spin()

if __name__ == '__main__':
	rospy.init_node('image_compensation')
	node = ImageCompensation()
	node.main()