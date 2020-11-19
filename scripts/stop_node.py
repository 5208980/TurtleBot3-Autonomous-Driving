#!/usr/bin/env python

import os
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header, Float64
from project_2.msg import Timer # Header, Float64

class StopNode:
	def __init__(self):
		self.onload()
		self.name = 'StopSignDetectNode ::'
		self.bridge = CvBridge()
		self.counter = 1

		self.sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.stop_sign_detect_cb, queue_size=1)
		# self.sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.stop_sign_detect_cb, queue_size=1)

		self.pub_timer = rospy.Publisher('/stop/timer', Timer, queue_size=1)	# 
		# self.pub_intersection = rospy.Publisher('/detect/intersection', UInt8, queue_size=1)
		self.pub_img = rospy.Publisher('/stop/image', Image, queue_size=1)	# Image with rectange
		self.pub_bot = rospy.Publisher('/stop/turtlebot', Image, queue_size=1)	

		self.timer = 0
		self.publish_stop_timer() # init timer msg
		self.cooldown = 0
		self.is_at_stop = False
		self.is_at_intersection = False

	def onload(self):
		# Load Classifier
		dir_path = os.path.dirname(os.path.realpath(__file__))
		dir_path += '/stop_data.xml'

		self.model = cv2.CascadeClassifier()
		if not self.model.load(dir_path):
			rospy.logerr('%s Classifier couldn\'t load', self.name)

	# Default bgr8 encoding
	def convert_compressed_image_to_cv(self, msg):
		frame_arr = np.fromstring(msg.data, np.uint8)
		frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
		return frame

	def mask_frame_for_intersection(self, frame):
		mask = np.zeros_like(frame[:,:,0])
		gap = 50
		polygon = np.array([ [0,frame.shape[0]], [0,frame.shape[0]-gap], [640,frame.shape[0]-gap], [640,frame.shape[0]] ])
		cv2.fillConvexPoly(mask, polygon, 1)
		frame = cv2.bitwise_and(frame, frame, mask=mask)
		
		return frame

	def hough_lines(self, frame):
		# Process
		'''
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.blur(frame,(5,5))
		_, frame = cv2.threshold(frame, 200 ,255, cv2.THRESH_BINARY)
		frame = cv2.blur(frame,(5,5))
		v = np.median(frame)
		sigma = 0.33
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))

		edges = cv2.Canny(frame, lower, upper, apertureSize = 3)
		'''

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur1 = cv2.blur(gray,(5,5))
		_, th_img = cv2.threshold(blur1,130,255,cv2.THRESH_BINARY)
		blur = cv2.blur(th_img,(5,5))

		self.pub_img.publish(self.bridge.cv2_to_imgmsg(th_img, "passthrough"))

		v = np.median(gray)
		sigma = 0.33
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edges = cv2.Canny(blur, lower, upper, apertureSize = 3)

		# lines = cv2.HoughLines(edges, 1, np.pi/90, 500)
		lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

		# lines =  cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
		return lines

	def bird_eye_perspective_transform(self, frame):
		top_x = 250
		top_y = 0
		bottom_x = 250
		bottom_y = 239

		pts_src = np.array([
			[320 - top_x, 360 - top_y],
			[320 + top_x, 360 - top_y],
			[320 + bottom_x, 240 + bottom_y],
			[320 - bottom_x, 240 + bottom_y]])
		pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])

		homo, status = cv2.findHomography(pts_src, pts_dst)
		warped = cv2.warpPerspective(frame, homo, (1000, 600))
		return warped

	# frame -> bool
	def intersection_detect(self, frame):
		frame_full = frame.copy()
		# frame = self.bird_eye_perspective_transform(frame)
		frame = self.mask_frame_for_intersection(frame)

		lines = self.hough_lines(frame)
		
		print(lines)
		intersection_exist = False
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

				if -0.2 < slope and slope < 0.2: # Horizontal Lines
					print("INTERSECT")
					cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
					intersection_exist = True

		# self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
		return intersection_exist

	# ===== TurtleBot Detection =====
	# Masking ROI
	def turtlebot_masking(self, frame):
		x = 0 + 50
		y = int((frame.shape[0]/2))
		h = 200
		w = frame.shape[1] - 50
		roi = frame[y:y+h, x:x+w]

		frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
		_, frame = cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY_INV)
		return frame

	# Most dominant colour in frame
	def unique_count_app(self, a):
		colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
		return colors[count.argmax()]

	# If there is TurtleBot in front of it
	def turtlebot_detection(self, frame):
		frame = self.turtlebot_masking(frame)
		colour = self.unique_count_app(frame)
		# print(colour[colour.argmax()])
		self.pub_bot.publish(self.bridge.cv2_to_imgmsg(frame, "passthrough"))
		if np.argmax(colour) >= 255:
			print(np.argmax(colour))
			return True
		return False
	# ================================

	def publish_stop_timer(self):
		h = Header()
		h.stamp = rospy.Time.now()

		t = Float64()
		t = self.timer

		int_msg = Timer()
		int_msg.header = h
		int_msg.data = t

		self.pub_timer.publish(int_msg)

	def stop_sign_detect_cb(self, msg):
		self.publish_stop_timer()
		# Drop frame for processing speed (6ps)
		if self.counter % 3 != 0:
			self.counter += 1
			return
		else:
			self.counter = 1

		frame = self.convert_compressed_image_to_cv(msg)
		
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		stop_signs = self.model.detectMultiScale(frame_gray, minSize=(20, 20))

		if self.timer > 0:
			# print(self.timer)
			if len(stop_signs) <= 0:
				self.timer -= 1
				self.publish_stop_timer()
			return

		if self.is_at_stop == True or self.is_at_intersection == True:
			self.cooldown = 5
			self.is_at_stop = False
			self.is_at_intersection = False

		if self.cooldown > 0:
			self.cooldown -= 1
			return

		# Use minSize because for not bothering with extra-small dots that would look like STOP signs
		intersections = self.intersection_detect(frame)	# bool

		self.turtlebot_detection(frame_rgb)

		if self.turtlebot_detection(frame_rgb):
			print("TurtleBot Detected Stopping ...")
			self.timer = 40
			self.publish_stop_timer()
			return

		if len(stop_signs) > 0:	# Found Stop Sign
			for (x, y, width, height) in stop_signs:
				cv2.rectangle(frame_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)
				
				if intersections:	# Found Intersection
					self.is_at_stop = True
					print("Stop Sign Stop")
					self.timer = 10
					self.publish_stop_timer()	
		else: # No Stop Sign
			if intersections: # Found Intersection
				self.is_at_intersection = True
				print("Intersection Stop")
				self.timer = 10
				self.publish_stop_timer()

		
		 # self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame_rgb, "rgb8"))
	def main(self):
		rospy.loginfo("%s Spinning", self.name)
		rospy.spin()

if __name__ == '__main__': # sys.argv
	rospy.init_node('stop_sign')
	node = StopNode()
	node.main()