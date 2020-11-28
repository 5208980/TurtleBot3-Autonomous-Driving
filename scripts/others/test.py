import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

class LaneDetection():
	def __init__(self):
		self.frame_rgb = None
		self.prev_left_fitx = None
		self.prev_right_fitx = None


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

		top_x = 210	# adjust top width
		top_y = 15  # adjust top height
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
		# plt.plot(frame)
		# plt.show()

		# Create an output image to draw on and visualize the result
		# out_img = np.dstack((frame, frame, frame)) * 255
		out_img = self.frame_rgb.copy()
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
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,255), 2)
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,0,255), 2)
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
		bad_lane = False
		if not len(leftx) is 0:
			left_fit = np.polyfit(lefty, leftx, 2)
			# print('Left lane coefficients : {}'.format(left_fit))

			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			# print(left_fitx[0])
			pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
			cv2.polylines(self.frame_rgb, np.int_([pts_left]), isClosed=False, color=(0, 255, 255), thickness=25)

			# if (not left_fitx is None) and (not self.prev_left_fitx is None):
			# 	prev_lane = np.mean(self.prev_left_fitx)
			# 	curr_lane = np.mean(left_fitx)
			# 	# # print('prev: {}'.format(np.mean(prev_lane)))
			# 	# # print('curr: {}'.format(np.mean(prev_lane)))
			# 	# diff_lane = (curr_lane - prev_lane)**2
			# 	# percentage = curr_lane - prev_lane
			# 	percentage = ((curr_lane - prev_lane)**2).mean(axis=None)
			# 	print("percentage: {}".format(percentage))
			#
			# 	if percentage == 0 or percentage > 5000:
			# 		print("BIG CHANGE")
			# 		bad_lane = True
			# 		left_fitx = self.prev_left_fitx

		right_fitx = None
		if not len(rightx) is 0:
			right_fit = np.polyfit(righty, rightx, 2)
			# print('Right lane coefficients : ')
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

			pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
			cv2.polylines(self.frame_rgb, np.int_([pts_right]), isClosed=False, color=(255, 0, 255), thickness=25)

			# if (not right_fitx is None) and (not self.prev_right_fitx is None):
			# 	prev_lane = np.mean(self.prev_right_fitx)
			# 	curr_lane = np.mean(right_fitx)
			# 	# # print('prev: {}'.format(np.mean(prev_lane)))
			# 	# # print('curr: {}'.format(np.mean(prev_lane)))
			# 	# diff_lane = (curr_lane - prev_lane)**2
			# 	# percentage = curr_lane - prev_lane
			# 	percentage = ((curr_lane - prev_lane)**2).mean(axis=None)
			# 	print("percentage: {}".format(percentage))
			#
			# 	if percentage == 0 or percentage > 5000:
			# 		print("BIG CHANGE")
			# 		bad_lane = True
			# 		right_fitx = self.prev_right_fitx


		left_x_mean = np.mean(leftx, axis=0)
		right_x_mean = np.mean(rightx, axis=0)
		lane_width = np.subtract(right_x_mean, left_x_mean)
		position = (10,50)
		# cv2.putText(self.frame_rgb, str(lane_width), position, cv2.FONT_HERSHEY_SIMPLEX, 1,(209, 80, 0, 255), 3)

		# # if lane is not in its half
		# if left_x_mean > 740 or right_x_mean < 740:
		# 	print("Lane: Wrong half")
		# 	left_fitx = self.prev_left_fitx
		# 	right_fitx = self.prev_right_fitx
		# 	# Set prev lanes instead of newly founded ones
		# 	# return self.prev_left_fitx, self.prev_right_fitx, None, None, self.frame_rgb

		# lane too small or large
		if (lane_width < 600 or lane_width > 800):
			print("Lane: Bad Size")
			left_fitx = self.prev_left_fitx
			right_fitx = self.prev_right_fitx
			# return self.prev_left_fitx, self.prev_right_fitx, None, None, self.frame_rgb

		# Draw Lanes
		if not left_fitx is None:
			pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
			cv2.polylines(self.frame_rgb, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)
		if not right_fitx is None:
			pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
			cv2.polylines(self.frame_rgb, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25)

		self.prev_left_fitx = left_fitx
		self.prev_right_fitx = right_fitx
		cv2.imshow('line', out_img)
		return left_fitx, right_fitx, left_lane_inds, right_lane_inds, self.frame_rgb	# bgr8
		# return None, None, None, None, out_img


	def image_cb(self, frame):
		frame = self.bird_eye_perspective_transform(frame)
		self.frame_rgb = frame.copy()

		frame = self.create_threshold_binary_image(frame)
		left_poly, right_poly, left_pts, right_pts ,frame = self.sliding_window(frame)

		if (not left_poly is None) and (not right_poly is None):
			centerx = np.mean([left_poly, right_poly], axis=0)
			# print(centerx.item(350))

		return frame

	def main(self):
		frame = cv2.imread('4_1_1.jpg')
		out = self.image_cb(frame)
		# cv2.imshow('line', self.frame_rgb)

		# cap = cv2.VideoCapture('output.mp4')
		#
		# while(cap.isOpened()):
		# 	ret, frame = cap.read()
		# 	if ret:
		# 		out = self.image_cb(frame)
		# 		cv2.imshow('line', out)
		# 		time.sleep(0.005)
		#
		# 	if cv2.waitKey(1) & 0xFF == ord('q'):
		# 		break
		# 	if cv2.waitKey(1) == ord('p'):
		# 		cv2.waitKey(-1) #wait until any key is pressed
		#
		cv2.waitKey(10000)
		cv2.destroyAllWindows()

if __name__ == '__main__': # sys.argv
	node = LaneDetection()
	node.main()

	# frame = cv2.imread('lane1.png')
	# image_cb(frame)
