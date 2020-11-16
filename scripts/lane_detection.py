#!/usr/bin/env python

from __future__ import division
import numpy as np
import math
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header, Float64
from project_2.msg import Lane	# Header, Float64

class LaneDetection:
	def __init__(self):
		self.name = "LaneDetection ::"
		self.frame_bgr = None
		self.prev_left_lane = None
		self.prev_right_lane = None
		self.bridge = CvBridge()

		self.bad_lanes = False
		self.prev_left_fitx = None
		self.prev_right_fitx = None

		# self.sub_lane = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_cb, queue_size=1)
		self.sub_lane = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_cb, queue_size=1)
		# self.sub_lane = rospy.Subscriber('/tranform/binary', Image, self.image_cb, queue_size=1)

		self.pub_bird_eye = rospy.Publisher('/transform/bird_eye', Image, queue_size=1)
		self.pub_binary = rospy.Publisher('/transform/binary', Image, queue_size=1)

		self.pub_center = rospy.Publisher('/detect/center', Lane, queue_size=1)
		self.pub_window = rospy.Publisher('/detect/window', Image, queue_size=1)
		self.pub_lane = rospy.Publisher('/detect/lane', Image, queue_size=1)

	# Default bgr8 encoding NOT CompressedImage
	def convert_image_to_cv(self, msg):
		frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		return frame

	def convert_compressed_image_to_cv(self, msg):
		frame_arr = np.fromstring(msg.data, np.uint8)
		frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
		return frame

	# image in black or white
	def create_threshold_binary_image(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.blur(frame,(5,5))
		_, frame = cv2.threshold(frame, 175 ,255, cv2.THRESH_BINARY)
		binary = cv2.blur(frame,(5,5))
		return binary

	# perspective transform (bird-eye view)
	def bird_eye_perspective_transform(self, frame):
		#top_x = 170
		#top_y = 30
		#bottom_x = 300
		#bottom_y = 230

		top_x = 210	# adjust top width
		top_y = 30  # adjust top height
		bottom_x = 360	# adjust bot width
		bottom_y = 237	# adjust bot height

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

		# frame for visualising sliding window
		sliding_window_frame = self.frame_bgr.copy()

		# Found Pecks of histogram. These two peaks represent the lanes
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
			cv2.rectangle(sliding_window_frame,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(2,255,255), 2)
			cv2.rectangle(sliding_window_frame,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(2,255,255), 2)
			
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

		left_fitx, right_fitx = None, None
		if not len(leftx) is 0:
			left_fit = np.polyfit(lefty, leftx, 2)
			# print('Left lane coefficients: {}'.format(left_fit))
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			# left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty + left_fit[3] 

			pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
			cv2.polylines(self.frame_bgr, np.int_([pts_left]), isClosed=False, color=(255, 0, 255), thickness=25)

		if not len(rightx) is 0:
			right_fit = np.polyfit(righty, rightx, 2)
			# print('Right lane coefficients: {}'.format(right_fit))
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
			# right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty + right_fit[3]

			pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
			cv2.polylines(self.frame_bgr, np.int_([pts_right]), isClosed=False, color=(255, 0, 255), thickness=25)
		
		# Validate lines
		if (not len(leftx) is 0) and (not len(rightx) is 0):
			left_x_mean = np.mean(leftx, axis=0)
			right_x_mean = np.mean(rightx, axis=0)
			lane_width = np.subtract(right_x_mean, left_x_mean)
			
			if left_x_mean > 740:
				# print("Bad Left Lane")
				# print(left_x_mean)
				left_fitx = None
			if right_x_mean < 740:
				# print("Bad Right Lane")
				# print(right_x_mean)
				right_fitx = None
				# return self.prev_left_fitx, self.prev_right_fitx, None, None, self.frame_bgr

			if lane_width < 100 or lane_width > 900:
				left_fitx = None

			# Draw width lane on image
			position = (10,50)
			cv2.putText(self.frame_bgr, str(lane_width), position, cv2.FONT_HERSHEY_SIMPLEX, 1,(209, 80, 0, 255), 3)
		elif (not len(leftx) is 0) and (len(rightx) is 0):
			left_fitx = None
		else:
			right_fitx = None

		self.bad_lanes = False

		# Sliding Window Image
		self.pub_window.publish(self.bridge.cv2_to_imgmsg(sliding_window_frame, "bgr8"))
		self.prev_left_fitx = left_fitx
		self.prev_right_fitx = right_fitx
		return left_fitx, right_fitx, left_lane_inds, right_lane_inds, self.frame_bgr # bgr8
		# return None, None, sliding_window_frame

	def publish_center(self, center, lanes):
		h = Header()
		h.stamp = rospy.Time.now()

		t = Float64()
		t = center

		n = Float64()
		n = lanes

		lane_msg = Lane()
		lane_msg.header = h
		lane_msg.center = t
		lane_msg.lanes = n
		self.pub_center.publish(lane_msg)

	def image_cb(self, msg):
		frame = self.convert_compressed_image_to_cv(msg)

		frame = self.bird_eye_perspective_transform(frame)
		# Bird Eye Perspective Image
		self.pub_bird_eye.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
		
		self.frame_bgr = frame.copy()	# Copy of coloured image

		frame = self.create_threshold_binary_image(frame)
		# Bird Eye Perspective(Binary) Image
		self.pub_binary.publish(self.bridge.cv2_to_imgmsg(frame, "passthrough"))

		# Left Lane, Right Lane, _, _, Output Frame
		left_poly, right_poly, left_pts, right_pts, frame = self.sliding_window(frame)

		if (not left_poly is None) and (not right_poly is None): # Left and Right Lane
			# print("Two")
			centerx = np.mean([left_poly, right_poly], axis=0)
			self.publish_center(centerx.item(350), 2)
		elif (left_poly is None) and (not right_poly is None): # Only Right Lane
			# print("Right")
			centerx = np.subtract(right_poly, 320)
			self.publish_center(centerx.item(350), 1)
		elif (not left_poly is None) and (right_poly is None): # Only Left Lane
			# print("Left")
			centerx = np.add(left_poly, 320)
			self.publish_center(centerx.item(350), 1)

		# Lane Lines Image
		self.pub_lane.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

	def main(self):
		rospy.loginfo("%s Spinning", self.name)
		rospy.spin()

if __name__ == '__main__': # sys.argv
	rospy.init_node('lane_detection')
	node = LaneDetection()
	node.main()
