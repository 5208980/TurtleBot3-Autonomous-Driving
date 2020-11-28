import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

frame = cv2.imread('gazebo2.jpg')

top_x = 300	# adjust top width
top_y = 0  # adjust top height
bottom_x = 500	# adjust bot width
bottom_y = 237	# adjust bot height

cv_image_original = cv2.GaussianBlur(frame, (5, 5), 0)

frame = cv2.line(frame, (320 - top_x, 360 - top_y), (320 + top_x, 360 - top_y), (0, 0, 255), 1)
frame = cv2.line(frame, (320 - bottom_x, 240 + bottom_y), (320 + bottom_x, 240 + bottom_y), (0, 0, 255), 1)
frame = cv2.line(frame, (320 + bottom_x, 240 + bottom_y), (320 + top_x, 360 - top_y), (0, 0, 255), 1)
frame = cv2.line(frame, (320 - bottom_x, 240 + bottom_y), (320 - top_x, 360 - top_y), (0, 0, 255), 1)

# homography transform process
# selecting 4 points from the original image
pts_src = np.array([
	[320 - top_x, 360 - top_y],
	[320 + top_x, 360 - top_y],
	[320 + bottom_x, 240 + bottom_y],
	[320 - bottom_x, 240 + bottom_y]])

# selecting 4 points from image that will be transformed
pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])

# finding homography matrix
h, status = cv2.findHomography(pts_src, pts_dst)

# homography process
cv_image_homography = cv2.warpPerspective(cv_image_original, h, (1000, 600))
# crop_cv_image_homography = cv_image_homography[0:897, 116:883]

f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(frame)
f.add_subplot(1,2, 2)
plt.imshow(cv_image_homography)
plt.show(block=True)

plt.imshow(frame)
plt.imshow(cv_image_homography)
plt.show()
