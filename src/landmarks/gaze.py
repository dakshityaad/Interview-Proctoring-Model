"""
Gaze estimation from MediaPipe face landmarks.

The idea in plain English:
     1. MediaPipe gives us normalized coordinates for the iris and four points
         around each eye.
     2. We measure where the iris sits between the eye corners and eyelids.
     3. We divide by the eye dimensions so the result does not change when the
         face moves closer to or farther from the camera.
     4. We average both eyes into one horizontal and vertical gaze position.
"""

# MediaPipe landmark indices for the iris centers.
RIGHT_IRIS, LEFT_IRIS = 468, 473

# Each eye uses its inner and outer corners for horizontal position, and its
# top and bottom eyelid points for vertical position.
RIGHT_EYE = {"inner": 133, "outer": 33, "top": 159, "bottom": 145}
LEFT_EYE = {"inner": 362, "outer": 263, "top": 386, "bottom": 374}


def get_gaze_ratio(landmarks, frame_width, frame_height):
    """Return the average horizontal and vertical iris positions.

    The returned values are normalized within each eye: 0.5 means the iris is
    centered, while values toward 0 or 1 indicate movement toward an edge.
    """
    def eye_ratio(iris_id, eye):
        # Convert normalized MediaPipe coordinates to pixels before measuring.
        iris_x = landmarks[iris_id].x * frame_width
        iris_y = landmarks[iris_id].y * frame_height
        inner_x = landmarks[eye["inner"]].x * frame_width
        outer_x = landmarks[eye["outer"]].x * frame_width
        top_y = landmarks[eye["top"]].y * frame_height
        bottom_y = landmarks[eye["bottom"]].y * frame_height

        horizontal_span = outer_x - inner_x
        vertical_span = bottom_y - top_y
        if abs(horizontal_span) < 1e-6 or abs(vertical_span) < 1e-6:
            # A collapsed eye cannot produce a meaningful ratio.
            return 0.5, 0.5

        # Dividing by the eye span cancels the effect of image scale.
        return (
            (iris_x - inner_x) / horizontal_span,
            (iris_y - top_y) / vertical_span,
        )

    # Combine the two independent eye measurements into one gaze estimate.
    right = eye_ratio(RIGHT_IRIS, RIGHT_EYE)
    left = eye_ratio(LEFT_IRIS, LEFT_EYE)
    return (right[0] + left[0]) / 2, (right[1] + left[1]) / 2