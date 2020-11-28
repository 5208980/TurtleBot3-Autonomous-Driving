import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# If given one lane then hog that one line (like wallfollow)

def average_line(lines):
	x1 ,x2 = 0, 100000
	y1, y2 = 0, 0
	if lines['n'] != 0:
		m = lines['slope']/lines['n']
		b = lines['intercept']/lines['n']

		y1, y2 = int(m*x1 + b), int(m*x2 + b)

		# print('y = {}*x + {}'.format(m, b))
		# print(x1, y1)
		# print(x2, y2)

	return (x1, y1), (x2, y2)

# Frame Masking for Area of interest (lane in front of TurtleBot)
# Frame.shape (640 x 480)
def maskFrame(frame):
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Creting Mask
	mask = np.zeros_like(frame[:,:,0])
	# polygon = np.array([[0,480], [0,400],  [160,240], [480,240], [640,400], [640,480]])
	polygon = np.array([[0,480], [0,400], [320, 240], [640,400], [640,480]])
	cv2.fillConvexPoly(mask, polygon, 1)
	# plt.imshow(mask, cmap= "gray")

	# Apply mask to get Area of interest
	img = cv2.bitwise_and(frame, frame, mask=mask)
	frame_copy = img.copy()

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = cv2.blur(gray_img,(5,5))
	_, img = cv2.threshold(img, 130 ,255, cv2.THRESH_BINARY)
	img = cv2.blur(img,(5,5))

	# return img

	v = np.median(gray_img)
	sigma = 0.33
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edges = cv2.Canny(img, lower, upper, apertureSize = 3)

	# edges = cv2.Canny(img, 50, 150, apertureSize = 3)
	# edges = cv2.Canny(gray, 180, 255, 30)

	'''
		Notes:
		np.pi/90 (to detect horizontal lines)
	'''
	lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
	# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=200)

	left_lane = { 'n': 0, 'slope': 0, 'intercept': 0 }
	right_lane = { 'n': 0, 'slope': 0, 'intercept': 0 }
	horizontal_lane = { 'n': 0, 'slope': 0, 'intercept': 0 }
	if not lines is None:
		for line in lines:
			rho, theta = line[0][0], line[0][1]
			a, b = np.cos(theta), np.sin(theta)
			x0, y0 = a*rho, b*rho
			x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
			x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))

			slope = (y2 - y1) / (x2 - x1)
			intercept = y1 - (slope * x1)

			# Horizontal lines slops between -0.2 < x < 0.2
			# if slope == -0.5092644581695677:
			# 	continue
			if -0.2 < slope and slope < 0.2:
				# Horizontal Lines
				print('horizontal lines')
				cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
			elif slope < 0:
				# Left or Right Lane
				print('Lane 1: {}'.format(slope))
				left_lane['n'] += 1
				left_lane['slope'] += slope
				left_lane['intercept'] += intercept
				cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2) # frame
			else:
				# Left or Right Lane
				print('Lane 2: {}'.format(slope))
				right_lane['n'] += 1
				right_lane['slope'] += slope
				right_lane['intercept'] += intercept

	pt1, pt2 = average_line(left_lane)
	cv2.line(frame_copy, pt1, pt2, (0, 0, 255), 2)

	# pt1, pt2 = average_line(right_lane)
	# cv2.line(frame_copy, pt1, pt2, (0, 0, 255), 2)

	# With Average Left and Right Lanes, We can Average Center Line for Turtlebot to move
	# If no Lanes, then the horizontal will come in play ...

	# cv2.imshow('line', frame_copy) # frame
	return frame_copy


def main():
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			out = maskFrame(frame)
			cv2.imshow('line', out)
			time.sleep(0.005)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__': # sys.argv
	cap = cv2.VideoCapture('output.mp4')
	main()

	# frame = cv2.imread('lane1.png')
	# out = maskFrame(frame)
	# plt.imshow(out)
	# plt.show()
