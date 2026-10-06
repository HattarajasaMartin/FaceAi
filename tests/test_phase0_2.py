from src.config import parse_person_name, resolve_name, status_from_count
from src.gallery import average_embeddings
import numpy as np
import unittest


class ConfigTests(unittest.TestCase):
    def test_resolve_anies_alias(self):
        self.assertEqual(resolve_name("Anis"), "anies")
        self.assertEqual(resolve_name("ANIES"), "anies")

    def test_accept_custom_name(self):
        slug, display = parse_person_name("Budi Santoso")
        self.assertEqual(slug, "budi_santoso")
        self.assertEqual(display, "Budi Santoso")
        self.assertEqual(resolve_name("budi"), "budi")

    def test_reject_empty_name(self):
        with self.assertRaises(ValueError):
            parse_person_name(" ")
        with self.assertRaises(ValueError):
            parse_person_name("x")

    def test_status_counts(self):
        self.assertEqual(status_from_count(0), "belum daftar")
        self.assertEqual(status_from_count(4), "belum lengkap")
        self.assertEqual(status_from_count(5), "terdaftar")
        self.assertEqual(status_from_count(10), "terdaftar")


class DetectLogicTests(unittest.TestCase):
    def test_face_count_messages(self):
        from src.detect import validate_face_count

        ok, msg = validate_face_count(0)
        self.assertFalse(ok)
        self.assertIn("tidak terdeteksi", msg.lower())

        ok, msg = validate_face_count(2)
        self.assertFalse(ok)
        self.assertIn("tepat satu", msg.lower())

        ok, msg = validate_face_count(1)
        self.assertTrue(ok)


class EmbeddingTests(unittest.TestCase):
    def test_average_is_normalized(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        mean = average_embeddings([a, b])
        self.assertAlmostEqual(float(np.linalg.norm(mean)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
