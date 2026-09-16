from types import SimpleNamespace
import unittest

from src.landmarks.gaze import get_gaze_ratio


def make_landmarks(scale):
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for inner, outer, top, bottom, iris in (
        (133, 33, 159, 145, 468),
        (362, 263, 386, 374, 473),
    ):
        center_x, center_y = 0.5, 0.5
        landmarks[inner] = SimpleNamespace(x=center_x - 0.1 * scale, y=center_y)
        landmarks[outer] = SimpleNamespace(x=center_x + 0.1 * scale, y=center_y)
        landmarks[top] = SimpleNamespace(x=center_x, y=center_y - 0.05 * scale)
        landmarks[bottom] = SimpleNamespace(x=center_x, y=center_y + 0.05 * scale)
        landmarks[iris] = SimpleNamespace(x=center_x + 0.04 * scale, y=center_y + 0.01 * scale)
    return landmarks


class GazeRatioTests(unittest.TestCase):
    def test_gaze_ratio_is_invariant_to_face_scale(self):
        near = get_gaze_ratio(make_landmarks(1.0), 640, 480)
        far = get_gaze_ratio(make_landmarks(0.5), 640, 480)

        self.assertAlmostEqual(far[0], near[0])
        self.assertAlmostEqual(far[1], near[1])