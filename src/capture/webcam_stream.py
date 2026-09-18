"""Frame generators for live webcam and recorded-video input."""

import cv2


def read_webcam_frames(camera_index=0):
	"""Yield frames from a webcam one at a time."""
	capture = cv2.VideoCapture(camera_index)
	capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				break
			yield frame
	finally:
		capture.release()


def read_video_frames(path):
	"""Yield frames from a recorded video file one at a time."""
	capture = cv2.VideoCapture(str(path))
	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				break
			yield frame
	finally:
		capture.release()
