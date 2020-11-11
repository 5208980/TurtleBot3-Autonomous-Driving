#!/usr/bin/env python

from __future__ import division
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64

from geometry_msgs.msg import Twist

import numpy as np
import math


class LaneDetection():
	def __init__(self):
		self.name = "LaneDetectionNode ::"
		self.frame_rgb = None
		self.prev_left_lane = None
		self.prev_right_lane = None
		self.bridge = CvBridge()
		self.sub_lane = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_cb, queue_size=1)

		self.pub_center = rospy.Publisher('/detect/lane', Float64, queue_size=1)
		self.pub_binary = rospy.Publisher('/lane/image/binary', Image, queue_size=1)
		self.pub_bird_eye = rospy.Publisher('/lane/image/bird_eye', Image, queue_size=1)
		self.pub_lane = rospy.Publisher('/lane/image', Image, queue_size=1)

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
		top_x = 110
		top_y = 20
		bottom_x = 250
		bottom_y = 200

		pts_src = np.array([
			[320 - top_x, 360 - top_y],
			[320 + top_x, 360 - top_y],
			[320 + bottom_x, 240 + bottom_y],
			[320 - bottom_x, 240 + bottom_y]])
		pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])

		homo, status = cv2.findHomography(pts_src, pts_dst)
		warped = cv2.warpPerspective(frame, homo, (1000, 600))
		return warped

	# should take in binary_frame (black and white image)
	def sliding_window(self, frame):
		# Take a histogram of the bottom half of the image
		bottom_half_y = frame.shape[0]/2
		histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
		# plt.plot(frame)
		# plt.show()

		# Create an output image to draw on and visualize the result
		# out_img = np.dstack((frame, frame, frame)) * 255
		out_img = frame.copy()
		# return None, None, out_img

		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 20
		# Set height of windows
		window_height = np.int(frame.shape[0]/nwindows)

		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = frame.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 50
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = frame.shape[0] - (window+1)*window_height
			win_y_high = frame.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(2,255,255), 2)
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(2,255,255), 2)
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		# Fit a second order polynomial to each


		warp_zero = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
		color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

		ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])

		left_fitx = None
		if not len(leftx) is 0:
			left_fit = np.polyfit(lefty, leftx, 2)
			# print('Left lane coefficients : ')
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

			pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
			cv2.polylines(self.frame_rgb, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)
			# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
			#
			# left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
			# cv2.polylines(out_img, [left], False, (1,1,0), thickness=5)

		right_fitx = None
		if not len(rightx) is 0:
			right_fit = np.polyfit(righty, rightx, 2)
			# print('Right lane coefficients : ')
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

			pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
			cv2.polylines(self.frame_rgb, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25)

			# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
			# right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
			# cv2.polylines(out_img, [right], False, (1,1,0), thickness=5)

		# final = cv2.addWeighted(frame, 1, color_warp, 0.2, 0)
		# final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)

		# out_img = np.uint8(np.dstack((frame, frame, frame))*255)
		# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
		# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
		# self.frame_rgb = cv2.resize(out_img*255,(640,480))

		'''
		left_x_mean = np.mean(leftx, axis=0)
		right_x_mean = np.mean(rightx, axis=0)
		lane_width = np.subtract(right_x_mean, left_x_mean)

		if left_x_mean > 740 or right_x_mean < 740:
			print("Bad Lane")
			return None, None, None, None, self.frame_rgb

		if  lane_width < 300 or lane_width > 800:
			print("Bad Lane")
			return None, None, None, None, self.frame_rgb
		'''
		return left_fitx, right_fitx, left_lane_inds, right_lane_inds, self.frame_rgb	# bgr8
		# return None, None, out_img


	def image_cb(self, msg):
		frame_arr = np.fromstring(msg.data, np.uint8)
		frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

		frame = self.bird_eye_perspective_transform(frame)
		self.pub_bird_eye.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
		self.frame_rgb = frame.copy()

		frame = self.create_threshold_binary_image(frame)
		# self.pub_binary.publish(self.bridge.cv2_to_imgmsg(frame, "passthrough"))
		left_poly, right_poly, left_pts, right_pts ,frame = self.sliding_window(frame)

		if (not left_poly is None) and (not right_poly is None):
			centerx = np.mean([left_poly, right_poly], axis=0)
			# print(centerx.item(350))
			msg_desired_center = Float64()
			msg_desired_center.data = centerx.item(350)
			self.pub_center.publish(msg_desired_center)

		self.pub_lane.publish(self.bridge.cv2_to_imgmsg(self.frame_rgb, "bgr8"))
		# return frame

	def main(self):
		rospy.loginfo("%s Spinning", self.name)
		rospy.spin()

if __name__ == '__main__': # sys.argv
	rospy.init_node('detect_lane')
	node = LaneDetection()
	node.main()
