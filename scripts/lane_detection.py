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
		self.bridge = CvBridge()

		self.prev_centerx = 0

	 	self.sub_lane = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_cb, queue_size=1)
		# self.sub_lane = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_cb, queue_size=1)

		self.pub_bird_eye = rospy.Publisher('/transform/bird_eye', Image, queue_size=1)
		self.pub_binary = rospy.Publisher('/transform/binary', Image, queue_size=1)

		self.pub_center = rospy.Publisher('/detect/center', Lane, queue_size=1)
		self.pub_window = rospy.Publisher('/detect/window', Image, queue_size=1)
		self.pub_lane = rospy.Publisher('/detect/lane', Image, queue_size=1)

	# Image -> CV (bgr8)
	def convert_image_to_cv(self, msg):
		frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		return frame

	# CompressedImage -> CV (bgr8)
	def convert_compressed_image_to_cv(self, msg):
		frame_arr = np.fromstring(msg.data, np.uint8)
		frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
		return frame

	# frame in b/w
	def create_threshold_binary_image(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.blur(frame,(5,5))
		_, frame = cv2.threshold(frame, 133 ,255, cv2.THRESH_BINARY)
		binary = cv2.blur(frame,(5,5))
		kernel = np.ones((5,5),np.float32)/25
		binary = cv2.filter2D(binary,-1,kernel)
		return binary

	# perspective transform (bird-eye view) 
	def bird_eye_perspective_transform(self, frame):
		top_x = 210		# adjust top width
		top_y = 0  		# adjust top height
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

	# change brightness of frame (reduce noise from light)
	def adjust_brightness(self, frame, value=-50):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)
		v = cv2.add(v,value)
		v[v > 255] = 255
		v[v < 0] = 0
		final_hsv = cv2.merge((h, s, v))
		frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
		return frame

	# crops top part of frame (reduce noise from walls)
	def crop_frame(self, frame):
		x = 0
		y = 100	# Cut of the top by y
		h = frame.shape[0]	# Should be 600
		w = frame.shape[1]	# Should be 1000
		cropped = frame[y:y+h, x:x+w]
		return cropped	#(500, 1000, 3)

	# find avg grad of poly
	def lane_gradient(self, polyfit):	
		poly = np.poly1d(polyfit)	
		poly_deriv = np.polyder(poly)	
		# poly_deriv[0] = b
		# poly_deriv[1] = m
		
		x = int(self.frame_bgr.shape[0]/2)
		# m = poly_deriv(x)

		return poly_deriv[1] if (not poly_deriv[1] is None) else 0

	# should take in binary frame (b/w frame)
	def sliding_window(self, frame):
		bottom_half_y = frame.shape[0]/2
		histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)

		sliding_window_frame = self.frame_bgr.copy()

		# Found Pecks of histogram. These two peaks represent the lanes
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		nwindows = 20
		window_height = np.int(frame.shape[0]/nwindows)

		nonzero = frame.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		leftx_current = leftx_base
		rightx_current = rightx_base

		margin = 50
		minpix = 50

		left_lane_inds = []
		right_lane_inds = []

		for window in range(nwindows):
			win_y_low = frame.shape[0] - (window+1)*window_height
			win_y_high = frame.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin

			cv2.rectangle(sliding_window_frame,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(2,255,255), 2)
			cv2.rectangle(sliding_window_frame,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(2,255,255), 2)
			
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)

			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		warp_zero = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
		color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

		ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])

		left_fitx, right_fitx = None, None
		leftm, rightm = 0, 0
		if not len(lefty) is 0:
			if len(leftx) < 500:
				left_fit = np.polyfit(lefty, leftx, 1)
				left_fitx = left_fit[0]*ploty + left_fit[1]

				leftm = left_fit[0]
				pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
				cv2.polylines(self.frame_bgr, np.int_([pts_left]), isClosed=False, color=(0, 255, 255), thickness=25)
			else:
				left_fit = np.polyfit(lefty, leftx, 2)
				left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

				leftm = self.lane_gradient(left_fit)
				pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
				cv2.polylines(self.frame_bgr, np.int_([pts_left]), isClosed=False, color=(0, 255, 255), thickness=25)

		if not len(rightx) is 0:
			if len(rightx) < 500:
				right_fit = np.polyfit(righty, rightx, 1)
				right_fitx = right_fit[0]*ploty

				rightm = right_fit[0]
				pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
				cv2.polylines(self.frame_bgr, np.int_([pts_right]), isClosed=False, color=(255, 255, 255), thickness=25)
			else:
				right_fit = np.polyfit(righty, rightx, 2)
				right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

				rightm = self.lane_gradient(right_fit)
				pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
				cv2.polylines(self.frame_bgr, np.int_([pts_right]), isClosed=False, color=(255, 255, 255), thickness=25)
		
		# Validate lines
		if (not len(leftx) is 0) and (not len(rightx) is 0):
			left_x_mean = np.mean(leftx, axis=0)
			right_x_mean = np.mean(rightx, axis=0)
			lane_width = np.subtract(right_x_mean, left_x_mean)

			if leftm > 0 and rightm > 0:	# Needs to turn right
				# print("Bad Right Lane")
				right_fitx = None
			elif leftm < 0 and rightm < 0:	# Needs to turn left
				# print("Bad Left Lane")
				left_fitx = None

			'''
			if lane_width < 200:	# Lane too small, move straight
				print("Lane Size!")
				right_fitx = None
				left_fitx = None
			# Draw width lane on image
			position = (10,50)
			cv2.putText(self.frame_bgr, str(lane_width), position, cv2.FONT_HERSHEY_SIMPLEX, 1,(209, 80, 0, 255), 3)
			'''

		self.pub_window.publish(self.bridge.cv2_to_imgmsg(sliding_window_frame, "bgr8")) # /detect/window
		return left_fitx, right_fitx, self.frame_bgr # bgr8


	def publish_center(self, center, lanes):
		self.prev_centerx = center
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

		# frame = self.adjust_brightness(frame, -5)
		frame = self.bird_eye_perspective_transform(frame)
		self.frame_bgr = frame.copy()	# Save original frame
		self.pub_bird_eye.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8")) # /trannsform/bird_eye
		frame = self.create_threshold_binary_image(frame)
		frame = self.crop_frame(frame)

		self.pub_binary.publish(self.bridge.cv2_to_imgmsg(frame, "passthrough"))  # /trannsform/binary

		# Left Lane, Right Lane, Frame
		left_poly, right_poly, frame = self.sliding_window(frame)

		if (not left_poly is None) and (not right_poly is None): # Left and Right Lane
			# print("Two")
			centerx = np.mean([left_poly, right_poly], axis=0)
			self.publish_center(centerx.item(350), 2)
			frame = cv2.circle(frame, (int (centerx.item(350)), 150),5, (0, 0, 0), 5)
		elif (left_poly is None) and (not right_poly is None): # Only Right Lane
			# print("Right")
			centerx = np.subtract(right_poly, 320)
			# print(centerx.item(350))
			self.publish_center(centerx.item(350), 1)
			frame = cv2.circle(frame, (int (centerx.item(350)), 150),5, (0, 0, 0), 5)
		elif (not left_poly is None) and (right_poly is None): # Only Left Lane
			# print("Left")
			centerx = np.add(left_poly, 320)
			self.publish_center(centerx.item(350), 1)
			frame = cv2.circle(frame, (int (centerx.item(350)), 150),5, (0, 0, 0), 5)
		else:
			# print("None")
			self.publish_center(self.prev_centerx, 0)

		self.pub_lane.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))	# /detect/lane

	def main(self):
		rospy.loginfo("%s Spinning", self.name)
		rospy.spin()

if __name__ == '__main__': # sys.argv
	rospy.init_node('lane_detection')
	node = LaneDetection()
	node.main()
