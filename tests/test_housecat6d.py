"""Reader and pose-metric regression tests without GPU or dataset downloads."""
import pickle
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from housecat6d import HouseCat6DScene, pose_errors
from Utils import _pixel_grid, depth2xyzmap_batch


class HouseCatTests(unittest.TestCase):
    def make_scene(self, root):
        for directory in ('rgb', 'depth', 'instance', 'labels', 'mesh'):
            (root / directory).mkdir()
        np.savetxt(root / 'intrinsics.txt', np.eye(3))
        for frame in ('000000', '000001'):
            rgb = np.zeros((3, 4, 3), dtype=np.uint8)
            rgb[..., 2] = 255
            cv2.imwrite(str(root / 'rgb' / f'{frame}.png'), rgb)
            cv2.imwrite(str(root / 'depth' / f'{frame}.png'), np.full((3, 4), 500, np.uint16))
            mask = np.full((3, 4), 255, np.uint8)
            mask[0, :2] = 7
            cv2.imwrite(str(root / 'instance' / f'{frame}.png'), mask)
            # Per-object amodal masks must not be mistaken for extra frames.
            cv2.imwrite(str(root / 'instance' / f'{frame}_cup-test.png'), mask)
            labels = dict(model_list=['cup-test'], instance_ids=[7], class_ids=[4],
                          rotations=np.eye(3)[None], translations=np.array([[0, 0, 0.5]]),
                          scales=np.array([0.2]))
            with (root / 'labels' / f'{frame}_label.pkl').open('wb') as stream:
                pickle.dump(labels, stream)
        (root / 'mesh' / 'cup-test.obj').write_text('test')

    def test_frame_ids_depth_rgb_masks_and_pose_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root)
            scene = HouseCat6DScene(root)
            self.assertEqual(len(scene), 2)
            rgb, depth, mask, labels = scene.frame(0)
            np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])
            np.testing.assert_allclose(depth, 0.5)
            self.assertEqual(scene.mask_for_instance(mask, 7).sum(), 2)
            records = list(scene.object_records(labels))
            self.assertEqual(records[0]['instance_id'], 7)
            np.testing.assert_allclose(records[0]['pose_gt'][:3, 3], [0, 0, 0.5])
            # scales is not an extra multiplier for metric OBJ coordinates.
            np.testing.assert_allclose(records[0]['pose_gt'][:3, :3], np.eye(3))
            self.assertTrue(scene.mesh_path('cup-test').is_file())
            with self.assertRaises(FileNotFoundError):
                scene.mesh_path('missing')

    def test_rejects_mismatched_frame_names_with_equal_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root)
            (root / 'depth' / '000001.png').rename(root / 'depth' / '000002.png')
            with self.assertRaisesRegex(ValueError, 'frame IDs'):
                HouseCat6DScene(root)

    def test_rejects_duplicate_instance_ids(self):
        labels = dict(model_list=['a', 'b'], instance_ids=[1, 1], class_ids=[1, 1],
                      rotations=np.repeat(np.eye(3)[None], 2, axis=0),
                      translations=np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            list(HouseCat6DScene.object_records(labels))

    def test_pose_error_units_and_rotation(self):
        gt = np.eye(4)
        pred = np.eye(4)
        pred[0, 3] = 0.01
        errors = pose_errors(pred, gt, np.array([[1, 0, 0], [0, 1, 0]]))
        self.assertAlmostEqual(errors['add_error_m'], 0.01)
        self.assertAlmostEqual(errors['translation_error_m'], 0.01)
        self.assertAlmostEqual(errors['rotation_error_deg'], 0)
        pred[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.assertAlmostEqual(pose_errors(pred, gt, np.zeros((1, 3)))['rotation_error_deg'], 90)

    def test_cached_xyz_grid_matches_camera_projection(self):
        depth = np.full((1, 2, 3), 0.5, dtype=np.float32)
        K = np.eye(3, dtype=np.float32)[None]
        actual = depth2xyzmap_batch(__import__('torch').from_numpy(depth),
                                    __import__('torch').from_numpy(K), np.inf)
        expected = np.array([[[[0., 0., .5], [0.5, 0., .5], [1., 0., .5]],
                              [[0., .5, .5], [.5, .5, .5], [1., .5, .5]]]], dtype=np.float32)
        np.testing.assert_allclose(actual.numpy(), expected)
        first = _pixel_grid(2, 3, 'cpu')
        second = _pixel_grid(2, 3, 'cpu')
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])


if __name__ == '__main__':
    unittest.main()
