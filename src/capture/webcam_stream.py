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


def get_video_fps(path):
	"""Return the recorded frame rate for a video, or fail clearly if absent."""
	capture = cv2.VideoCapture(str(path))
	try:
		fps = float(capture.get(cv2.CAP_PROP_FPS))
	finally:
		capture.release()
	if fps <= 0:
		raise ValueError(f"Could not determine FPS for video: {path}")
	return fps
