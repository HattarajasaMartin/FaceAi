from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.camera import CameraError, open_camera
from src.config import LOCKOUT_LIMIT, score_percent
from src.detect import validate_face_count
from src.domain import FaceHit
from src.gesture import FingerState
from src.lockout import LockoutStore
from src.overlay import employee_caption, pretty_name
from src.quality import assess_quality, liveness_from_yaw


def _face(image_size: int = 320, box_size: int = 140, yaw: float = 0.0, x: int = 90, y: int = 90) -> FaceHit:
    right_eye = x + box_size * 0.3
    left_eye = x + box_size * 0.7
    nose = (right_eye + left_eye) / 2 + yaw * abs(left_eye - right_eye)
    raw = np.array(
        [
            x, y, box_size, box_size,
            right_eye, y + 40,
            left_eye, y + 40,
            nose, y + 70,
            x + 40, y + 110,
            x + 100, y + 110,
            0.95,
        ],
        dtype=np.float32,
    )
    return FaceHit(box=(x, y, box_size, box_size), score=0.95, raw=raw)


class GestureTests(unittest.TestCase):
    def test_peace_sign_two_fingers(self):
        peace = FingerState(True, True, False, False)
        self.assertTrue(peace.peace)
        self.assertFalse(peace.one_finger)
        self.assertEqual(peace.gesture, "peace")
        self.assertFalse(FingerState(True, True, True, False).peace)

    def test_three_fingers(self):
        three = FingerState(True, True, True, False)
        self.assertTrue(three.three_fingers)
        self.assertFalse(three.peace)
        self.assertEqual(three.gesture, "three")

    def test_one_finger(self):
        one = FingerState(True, False, False, False)
        self.assertTrue(one.one_finger)
        self.assertFalse(one.peace)
        self.assertEqual(one.gesture, "one")


class CaptionTests(unittest.TestCase):
    def test_enrolled_name_shows_company(self):
        title, role = employee_caption("prabowo")
        self.assertEqual(title, "Prabowo")
        self.assertEqual(role, "Karyawan PT. Testing")

    def test_unknown_face(self):
        title, role = employee_caption(None)
        self.assertEqual(title, "Tidak dikenali")
        self.assertEqual(role, "PT. Testing")

    def test_pretty_name_title_case(self):
        self.assertEqual(pretty_name("anies"), "Anies")
        self.assertEqual(pretty_name("Dedi Mulyadi"), "Dedi Mulyadi")

    def test_score_percent(self):
        self.assertEqual(score_percent(0.87), 87)
        self.assertEqual(score_percent(1.0), 100)
        self.assertEqual(score_percent(0.0), 0)


class FaceCountTests(unittest.TestCase):
    def test_zero_and_two_faces(self):
        ok, message = validate_face_count(0)
        self.assertFalse(ok)
        self.assertEqual(message, "Wajah tidak terdeteksi")
        ok, message = validate_face_count(2)
        self.assertFalse(ok)
        self.assertIn("Harus tepat satu wajah", message)


class QualityTests(unittest.TestCase):
    def test_dark_image_rejected(self):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        report = assess_quality(image, _face())
        self.assertFalse(report.ok)
        self.assertTrue(any("gelap" in item.lower() for item in report.errors))

    def test_blurry_flat_crop_rejected(self):
        image = np.full((320, 320, 3), 180, dtype=np.uint8)
        report = assess_quality(image, _face())
        self.assertFalse(report.ok)
        self.assertTrue(any("buram" in item.lower() for item in report.errors))

    def test_tilted_face_rejected(self):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        image[:] = (40, 180, 90)
        image[::3, ::3] = 255
        report = assess_quality(image, _face(yaw=0.9))
        self.assertTrue(any("miring" in item.lower() for item in report.errors))

    def test_small_face_rejected(self):
        image = np.full((320, 320, 3), 160, dtype=np.uint8)
        image[::2, ::2] = 20
        report = assess_quality(image, _face(box_size=40))
        self.assertTrue(any("kecil" in item.lower() for item in report.errors))

    def test_liveness_needs_head_motion(self):
        ok, _ = liveness_from_yaw([0.01, 0.02, 0.0], min_range=0.08)
        self.assertFalse(ok)
        ok, _ = liveness_from_yaw([-0.1, 0.12], min_range=0.08)
        self.assertTrue(ok)


class LockoutTests(unittest.TestCase):
    def test_lock_after_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LockoutStore(Path(tmp) / "lockout.json")
            for _ in range(LOCKOUT_LIMIT - 1):
                self.assertEqual(store.register_failure("anies"), 0)
            remaining = store.register_failure("anies")
            self.assertGreater(remaining, 0)
            store.register_success("anies")
            self.assertEqual(store.remaining_seconds("anies"), 0)


class CameraTests(unittest.TestCase):
    def test_unavailable_camera(self):
        class Closed:
            def isOpened(self):
                return False

            def release(self):
                return None

        with patch("src.camera.cv2.VideoCapture", return_value=Closed()):
            with self.assertRaises(CameraError):
                open_camera()


if __name__ == "__main__":
    unittest.main()
