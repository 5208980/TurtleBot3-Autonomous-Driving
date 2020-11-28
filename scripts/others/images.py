import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def bird_eye_perspective_transform(frame):
	top_x = 170
	top_y = 50
	bottom_x = 300
	bottom_y = 170

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

def create_threshold_binary_image(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame = cv2.blur(frame,(5,5))
	_, frame = cv2.threshold(frame, 133 ,255, cv2.THRESH_BINARY)
	binary = cv2.blur(frame,(5,5))
	return binary

def histogram(frame):
	bottom_half_y = frame.shape[0]/2
	histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
	return histogram

def turtlebot_masking(frame):
	x = 0 + 50
	y = int((frame.shape[0]/2))
	h = 200
	w = frame.shape[1] - 50
	roi = frame[y:y+h, x:x+w]

	frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
	_, frame = cv2.threshold(frame, 35, 255, cv2.THRESH_BINARY_INV)

	return frame

# Most dominant colour in frame
def unique_count_app(frame):
	histogram = cv2.calcHist([frame], [0], None, [256], [0, 256])
	#print(np.argmax(histogram))
	return np.argmax(histogram)

# If there is TurtleBot in front of it
def turtlebot_detection(frame):
	frame = self.turtlebot_masking(frame)
	colour = self.unique_count_app(frame)
	self.pub_bot.publish(self.bridge.cv2_to_imgmsg(frame, "passthrough"))
	if colour >= 250:
		# print(np.argmax(colour))
		print("TurtleBot Spotted")
		return True
	return False

def unique_count_app(a):
	colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
	return colors[count.argmax()]

if __name__ == '__main__': # sys.argv
	frame = cv2.imread('robot_front.png')

	frame = turtlebot_masking(frame)
	unique_count_app(frame)

	# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	# _, frame = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY_INV)
	# histogram = cv2.calcHist([frame], [0], None, [256], [0, 256])
	# print(np.argmax(histogram))
	# frame = turtlebot_masking(frame)
	# colour = unique_count_app(frame)

	# histogram = histogram(frame)
	# plt.plot(histogram)
	# plt.show()
	cv2.imshow('img', frame)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()
