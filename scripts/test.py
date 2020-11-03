import os
import cv2

self.sift = cv2.xfeatures2d.SIFT_create()

# get the stop sign image in assets
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('project-2/scripts', 'project-2/assets')
self.stop_sign_img = cv2.imread(dir_path + 'stop.jpg', 0)

# stop sign image features
self.kp2, self.des2 = self.sift.detectAndCompute(self.stop_sign_img, None)