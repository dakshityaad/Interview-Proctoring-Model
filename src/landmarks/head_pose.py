"""
Head pose estimation from MediaPipe face landmarks.

The idea in plain English:
    1. Every human face has the same rough shape - nose, chin, eye corners,
       mouth corners. We store the real-world coordinates of those 6 points
       for a generic face below (GENERIC_FACE_3D).
    2. MediaPipe finds those same 6 points in the webcam image (pixels).
    3. cv2.solvePnP asks: "if I take a generic face and rotate/move it in
       3D space, what rotation makes it line up with what the camera sees?"
    4. From that rotation we extract two friendly numbers:
          pitch -> looking up (+) or down (-)
          yaw   -> turned to their left (+) or right (-)
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# The 6 reference points
# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (fixed by MediaPipe's face model).
NOSE_TIP        = 1    # tip of the nose
CHIN            = 152  # bottom of the chin
LEFT_EYE_OUTER  = 263  # outer corner of the person's LEFT eye (right side of image)
RIGHT_EYE_OUTER = 33   # outer corner of the person's RIGHT eye (left side of image)
LEFT_MOUTH      = 287  # left corner of the mouth
RIGHT_MOUTH     = 57   # right corner of the mouth



# The 6 points above, listed in the SAME ORDER, as real-world 3D coordinates
# in millimeters on an "average" face, in the head's own frame:
#   +x = person's left, +y = up, +z = out of the face (toward the camera).
# These are the classic values from the well-known OpenCV head-pose tutorial.
# Faces vary, but solvePnP only needs them approximately right.
GENERIC_FACE_3D = np.array(
    [
        [0.0,     0.0,     0.0],    # nose tip
        [0.0,   -330.0,  -65.0],    # chin
        [-225.0,  170.0, -135.0],   # left eye, outer corner
        [225.0,   170.0, -135.0],   # right eye, outer corner
        [-150.0, -150.0, -125.0],   # left mouth corner
        [150.0,  -150.0, -125.0],   # right mouth corner
    ],
    dtype=np.float64,
)

# Convenience lookup: which landmark indices to pull out of the full
# 478-point face mesh, in the same order as GENERIC_FACE_3D.
POSE_LANDMARK_INDICES = np.array(
    [NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH],
    dtype=np.int32,
)


def get_head_pose(landmarks, frame_width, frame_height):
    """Estimate head pitch and yaw from one face's landmarks.

    Args:
        landmarks:     list of MediaPipe normalized face landmarks for ONE
                       face (each landmark has .x and .y in the 0..1 range).
        frame_width:   webcam frame width in pixels.
        frame_height:  webcam frame height in pixels.

    Returns:
        (pitch, yaw) in degrees, or (None, None) if the pose could not be
        solved (e.g. no landmarks were detected this frame).

        Sign convention:
            pitch > 0 -> head tilted UP,    pitch < 0 -> head tilted DOWN
            yaw   > 0 -> turned to THEIR left, yaw < 0 -> to THEIR right
    """
    # Bail out early if MediaPipe didn't find a face this frame.
    if not landmarks:
        return None, None

    # --- 1. Collect the 2D pixel positions of our 6 reference points ------
    # Landmarks come in as normalized values (0..1), so multiply by the
    # frame size to get actual pixel coordinates.
    image_points = np.array(
        [
            [landmarks[i].x * frame_width, landmarks[i].y * frame_height]
            for i in POSE_LANDMARK_INDICES
        ],
        dtype=np.float64,
    )

    # --- 2. Build a rough camera model ------------------------------------
    # We don't know the exact webcam lens parameters, so we assume a
    # "typical" camera: the optical center sits in the middle of the frame
    # and the focal length equals the frame width. This approximation is
    # good enough for estimating head angles.
    focal_length = frame_width
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1],
        ],
        dtype=np.float64,
    )
    # Assume no lens distortion.
    dist_coeffs = np.zeros((4, 1))

    # --- 3. Let OpenCV solve for the head's rotation ----------------------
    # solvePnP = "given where these points are on a real 3D face and where
    # they appear in the image, figure out how the head is rotated."
    success, rotation_vector, _ = cv2.solvePnP(
        GENERIC_FACE_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None

    # --- 4. Turn the rotation into a matrix we can read angles from -------
    # solvePnP gives us a rotation *vector* (3 numbers, compact but not
    # human-friendly). Rodrigues converts it to a plain 3x3 matrix so we
    # can pull the angles out with simple trigonometry.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # --- 5. Extract pitch and yaw from the matrix -------------------------
    # Using the standard rotation order (yaw -> pitch -> roll):
    #   pitch comes from the matrix element that holds "nose up/down"
    #   yaw   comes from the pair that holds "turned left/right"
    sin_pitch = -rotation_matrix[1, 2]

    # Safety clamp: floating-point noise can push sin_pitch slightly past
    # 1.0, and asin() would then return NaN, which would silently break
    # everything downstream.
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)

    pitch = np.degrees(np.arcsin(sin_pitch))
    yaw = np.degrees(
        np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
    )

    return float(pitch), float(yaw)