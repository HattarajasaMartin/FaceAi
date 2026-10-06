from __future__ import annotations

import unittest

import numpy as np

from src.domain import NameScore
from src.faceid import decide_faceid
from src.recognize_core import decide_recognition, score_against_gallery
from src.verify_core import decide_verify

SAMPLE_NAMES = ("prabowo", "anies", "jokowi", "mulyadi", "ahok")


def _one_hot(index: int, size: int = 8) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _gallery() -> dict[str, np.ndarray]:
    return {name: _one_hot(index) for index, name in enumerate(SAMPLE_NAMES)}


class RecognitionTests(unittest.TestCase):
    def test_match_anies(self):
        scores = score_against_gallery(_one_hot(1), _gallery())
        result = decide_recognition(scores)
        self.assertEqual(result.status, "cocok")
        self.assertEqual(result.best_name, "anies")
        self.assertGreater(result.best_score, 0.9)
        self.assertIn("100%", result.message)

    def test_unknown_below_threshold(self):
        unknown = np.ones(8, dtype=np.float32)
        unknown = unknown / np.linalg.norm(unknown)
        scores = score_against_gallery(unknown, _gallery())
        result = decide_recognition(scores, threshold=0.9)
        self.assertEqual(result.status, "tidak_dikenali")

    def test_empty_gallery(self):
        result = decide_recognition(())
        self.assertEqual(result.status, "tidak_dikenali")
        self.assertIn("Belum ada", result.message)


class FaceIdTests(unittest.TestCase):
    def test_known_face_accepted(self):
        scores = score_against_gallery(_one_hot(1), _gallery())
        result = decide_faceid(decide_recognition(scores))
        self.assertTrue(result.accepted)
        self.assertEqual(result.code, "TERIMA_FACEID")
        self.assertIn("100%", result.message)

    def test_unknown_face_rejected(self):
        unknown = np.ones(8, dtype=np.float32)
        unknown = unknown / np.linalg.norm(unknown)
        scores = score_against_gallery(unknown, _gallery())
        result = decide_faceid(decide_recognition(scores, threshold=0.9))
        self.assertFalse(result.accepted)
        self.assertEqual(result.code, "TOLAK_FACEID")
        self.assertIn("TOLAK", result.message)


class VerifyMatrixTests(unittest.TestCase):
    def _scores(self, face_name: str) -> tuple[NameScore, ...]:
        scores = []
        for name in SAMPLE_NAMES:
            scores.append(NameScore(name, 0.92 if name == face_name else 0.11))
        return tuple(sorted(scores, key=lambda item: item.score, reverse=True))

    def test_accept_only_matching_claim(self):
        enrolled = frozenset(SAMPLE_NAMES)
        for face in SAMPLE_NAMES:
            for claim in SAMPLE_NAMES:
                result = decide_verify(claim, self._scores(face), enrolled)
                if claim == face:
                    self.assertTrue(result.accepted, msg=f"{claim} vs {face}")
                    self.assertEqual(result.code, "TERIMA")
                else:
                    self.assertFalse(result.accepted, msg=f"{claim} vs {face}")
                    self.assertEqual(result.code, "TOLAK_IMPERSONATION")

    def test_unknown_person_rejected(self):
        enrolled = frozenset(SAMPLE_NAMES)
        scores = tuple(NameScore(name, 0.12) for name in SAMPLE_NAMES)
        result = decide_verify("anies", scores, enrolled)
        self.assertFalse(result.accepted)
        self.assertEqual(result.code, "TOLAK_MISMATCH")
        self.assertIn("tidak cocok", result.message.lower())

    def test_unenrolled_claim(self):
        scores = self._scores("anies")
        result = decide_verify("prabowo", scores, frozenset({"anies"}))
        self.assertEqual(result.code, "TOLAK_UNENROLLED")
        self.assertIn("belum terdaftar", result.message.lower())


if __name__ == "__main__":
    unittest.main()
