import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time

def load_model(name):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	dir_path += '/stop_data.xml'

	model = cv2.CascadeClassifier()
	if not model.load(dir_path):
		return None

	return model

if __name__ == '__main__': # sys.argv
	model = load_model('/stop_data.xml')

	frame = cv2.imread('stop1.png')
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	stop_signs = model.detectMultiScale(frame_gray, minSize=(20, 20))

	if len(stop_signs) > 0:	# Found Stop Sign
			for (x, y, width, height) in stop_signs:
				cv2.rectangle(frame, (x, y), (x + height, y + width),  (0, 255, 0), 5)
				
	cv2.imshow('img', frame)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()
