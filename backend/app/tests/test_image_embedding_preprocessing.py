from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from service.image_embedding_service import (
    IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
    IMAGE_EMBEDDING_PROFILE_NATIVE,
    prepare_image_embedding_input,
)


class ImageEmbeddingPreprocessingTest(unittest.TestCase):
    def test_native_profile_returns_original_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "icon.png"
            cv2.imwrite(str(path), np.zeros((2, 2, 4), dtype=np.uint8))

            result = prepare_image_embedding_input(
                path,
                preprocessing=IMAGE_EMBEDDING_PROFILE_NATIVE,
            )

            self.assertEqual(result, path)

    def test_alpha_white_profile_composites_transparency_without_new_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "icon.png"
            image = np.zeros((2, 2, 4), dtype=np.uint8)
            image[0, 0] = [10, 20, 30, 255]
            image[0, 1] = [10, 20, 30, 0]
            cv2.imwrite(str(path), image)

            result = prepare_image_embedding_input(
                path,
                preprocessing=IMAGE_EMBEDDING_PROFILE_ALPHA_WHITE,
            )

            self.assertIsNotNone(result)
            self.assertNotEqual(result, path)
            prepared = cv2.imread(str(result), cv2.IMREAD_UNCHANGED)
            self.assertEqual(prepared.shape[2], 3)
            self.assertTrue(np.array_equal(prepared[0, 0], [10, 20, 30]))
            self.assertTrue(np.array_equal(prepared[0, 1], [255, 255, 255]))


if __name__ == "__main__":
    unittest.main()
