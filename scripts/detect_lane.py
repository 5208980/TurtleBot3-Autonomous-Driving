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

class LaneDetectNode():
	def __init__(self):
		self.name = "LaneDetectionNode ::"
		self.counter = 0
		self.avgx = 0
		self.top_x = 250
		self.top_y = 5
		self.bottom_x = 250
		self.bottom_y = 239
		self.bridge = CvBridge()
		# self.sub_lane = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.detect_lane_cb, queue_size = 1)
		self.sub_lane = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.detect_lane_cb, queue_size=1)		
		self.pub_image = rospy.Publisher('/lane_detect/image', Image, queue_size=1)
		self.pub_image_right = rospy.Publisher('/lane_detect/image/right', Image, queue_size=1)
		self.pub_image_left = rospy.Publisher('/lane_detect/image/left', Image, queue_size=1)
		self.pub_action = rospy.Publisher('/lane_detect/action', Float64, queue_size=1)
		self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

		self.prev_action = 0
		self.frame_threshold = 0
		self.prev_frames = []

	# Calculate Cluster of lines in lanes
	def average_line(self, lines):
		if lines['n'] != 0:
			m = lines['slope']/lines['n']
			b = lines['intercept']/lines['n']
			# (Gradient, Y-Intercept)
			return (m, b)

		return None

	# Returns two point of given gradient and y-inter
	def get_points(self, m, b):
		x1 = int(0)
		x2 = int(480/2) # 100000
		y1 = int(m*x1 + b)
		y2 = int(m*x2 + b)
		return (x1, y1), (x2, y2)

	# Gets the x-intercept
	def get_x_intercept(self, m, b):
		x, y = 0, 0
		if m != 0:
			x = (-b)/m
		return (x, y)

	# Frame Masking for Area of interest (lane in front of TurtleBot)
	def mask_lanes(self, frame, arr):
		mask = np.zeros_like(frame[:,:,0])
		polygon = np.array(arr)
		cv2.fillConvexPoly(mask, polygon, 1)
		frame = cv2.bitwise_and(frame,frame,mask=mask)

		return frame

	# Change (Like Bird Eye) perception of image
	def projected_perspective(self, frame):
		# Vertices extracted manually for performing a perspective transform
		bottom_left = [105, 480]
		bottom_right = [530, 480]
		top_left = [205, 300]
		top_right = [435, 300]

		source = np.float32([bottom_left,bottom_right,top_right,top_left])

		# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
		bottom_left = [200, 480]
		bottom_right = [440, 480]
		top_left = [200, 1]
		top_right = [440, 1]

		dst = np.float32([bottom_left,bottom_right,top_right,top_left])
		M = cv2.getPerspectiveTransform(source, dst)
		warped_size = (640, 480)
		warped = cv2.warpPerspective(frame, M, warped_size, flags=cv2.INTER_NEAREST)
		
		# out_img = np.uint8(np.dstack((warped, warped, warped))*255)

		return warped
		
		'''
		pts_src = np.array([
		[320 - self.top_x, 360 - self.top_y],
		[320 + self.top_x, 360 - self.top_y],
		[320 + self.bottom_x, 240 + self.bottom_y],
		[320 - self.bottom_x, 240 + self.bottom_y]])

		pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])
		h, status = cv2.findHomography(pts_src, pts_dst)
		frame = cv2.warpPerspective(frame, h, (1000, 600))
		frame = frame[0:897, 116:883]	# remove the black sides

		return frame
		'''

	# Original -> GrayScale -> Darken -> HLS -> Threshold -> Gaussian Blur -> Canny -> Hough
	def detect_lines(self, frame):
		width, height = frame.shape[0],frame.shape[1]
		frame_copy = frame.copy()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # GrayScale
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # HSV

		lower = np.array([0,0,122])
		upper = np.array([175,24,255])
		mask = cv2.inRange(hsv, lower, upper)
		threshold = cv2.bitwise_and(frame_copy, frame_copy, mask= mask)
		blur = cv2.GaussianBlur(threshold, (5,5), cv2.BORDER_DEFAULT) # Gaussian
		# blur = cv2.GaussianBlur(blur, (5,5), cv2.BORDER_DEFAULT) # Gaussian

		_, filtered = cv2.threshold(blur, 133 ,255, cv2.THRESH_BINARY)
		# edges = cv2.Canny(blur, 50, 150)
		# return frame_filtered
		
		v = np.median(frame)
		sigma = 0.33
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edges = cv2.Canny(filtered, lower, upper, apertureSize = 3) # Canny
		
		lines = cv2.HoughLines(edges, 1, np.pi/180, 80) # Hough
		# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=200)
		# lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi/180, 90, minLineLength = 50, maxLineGap = 2)

		lane = { 'n': 0, 'slope': 0, 'intercept': 0 }
		if not lines is None:
			for line in lines:
				rho, theta = line[0][0], line[0][1]
				a, b = np.cos(theta), np.sin(theta)
				x0, y0 = a*rho, b*rho
				x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
				x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))

				if x2 - x1 == 0:
					slope = 0
				else:
					slope = (y2 - y1) / (x2 - x1)

				intercept = y1 - (slope * x1)

				if -0.2 < slope and slope < 0.2:
					cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
					x = 0
				else:
					lane['n'] += 1
					lane['slope'] += slope
					lane['intercept'] += intercept
					cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
					cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)

		# cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

		# mb[0], mb[1] = gradient, y-intercept
		mb = self.average_line(lane) 
		if not mb is None:
			lane = { 'n': 1, 'slope': mb[0], 'intercept': mb[1] }
			# CvFrame, {nLane, gradient, intercept}
			return frame, lane
		return frame, None

	def detect_and_draw_lane(self, frame):
		frame_copy = frame.copy()	# Original

		height, width, channel = frame.shape
		frame_left = self.mask_lanes(frame_copy, [[0, 0], [int(width/2), 0], [int(width/2), height], [0, height]])
		frame_right = self.mask_lanes(frame_copy, [[int(width/2), 0], [width, 0], [width, height], [int(width/2), height]])

		frame_right_with_lines, right_lane = self.detect_lines(frame_right)
		frame_left_with_lines, left_lane = self.detect_lines(frame_left)					
		# frame_left_with_lines, left_lane = None, None
		# frame_right_with_lines, right_lane = None, None
		# Used for Visualisation (Left and Right Lane Detection)
		
		
		if not right_lane is None:
			pt_r1, pt_r2 = self.get_points(right_lane['slope'], right_lane['intercept'])
			cv2.line(frame_right_with_lines, pt_r1, pt_r2, (2, 255, 255), 2)
		img_msg = self.bridge.cv2_to_imgmsg(frame_right_with_lines, "bgr8")
		self.pub_image_right.publish(img_msg)

		if not left_lane is None:
			pt_l1, pt_l2 = self.get_points(left_lane['slope'], left_lane['intercept'])
			cv2.line(frame_left_with_lines, pt_l1, pt_l2, (2, 255, 255), 2)
		img_msg = self.bridge.cv2_to_imgmsg(frame_left_with_lines, "bgr8")
		self.pub_image_left.publish(img_msg)
		
		# Found Robot Line
		if not (right_lane and left_lane) is None: # If Left and Right Available (Average Out Lines)
			print("Two Lanes")
			pt_r1, pt_r2 = self.get_points(right_lane['slope'], right_lane['intercept'])
			cv2.line(frame, pt_r1, pt_r2, (255, 255, 255), 2)
			pt_l1, pt_l2 = self.get_points(left_lane['slope'], left_lane['intercept'])
			cv2.line(frame, pt_l1, pt_l2, (0, 255, 255), 2)
		elif not left_lane is None: # If Left Available (Turn right)
			print("Left Mask")
			pt_r1, pt_r2 = self.get_points(left_lane['slope'], left_lane['intercept'])
			cv2.line(frame, pt_r1, pt_r2, (255, 255, 255), 2)
		elif not right_lane is None: # If Right Available (Turn left)
			print("Right Mask")
			pt_r1, pt_r2 = self.get_points(right_lane['slope'], right_lane['intercept'])
			print(right_lane['intercept'])
			cv2.line(frame, pt_r1, pt_r2, (255, 255, 255), 2)
		else: # If None (TravelStraight or takest last known center)
			print("No Lanes !!")
			cv2.line(frame, (self.avgx,0), (self.avgx,height), (255, 0, 0), 2)
			# self.prev_action = action

		return frame	

	def detect_lane_cb(self, frame):

		
		frame_arr = np.fromstring(frame.data, np.uint8)		# Convert image to Cv
		frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
		frame = self.projected_perspective(frame)			# Project image

		frame = self.detect_and_draw_lane(frame)	
		# frame = self.detect_lines(frame)	

		img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
		self.pub_image.publish(img_msg)	

	def main(self):
		rospy.loginfo("%s Spinning", self.name)
		rospy.spin()

if __name__ == '__main__': # sys.argv
	rospy.init_node('detect_lane')
	node = LaneDetectNode()
	node.main()
