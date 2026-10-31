"""HouseCat6D scene reader for multi-object FoundationPose evaluation."""

import pickle
from pathlib import Path

import cv2
import numpy as np


class HouseCat6DScene:
    """Read one HouseCat6D scene without a detector or an ID tracker."""

    def __init__(self, scene_dir, mesh_dir=None, depth_scale=1e-3):
        self.root = Path(scene_dir)
        self.mesh_dir = Path(mesh_dir) if mesh_dir else self.root / "mesh"
        self.depth_scale = depth_scale
        self.rgb_files = sorted((self.root / "rgb").glob("*.png"))
        self.depth_files = sorted((self.root / "depth").glob("*.png"))
        self.instance_files = sorted((self.root / "instance").glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
        self.label_files = sorted((self.root / "labels").glob("*_label.pkl"))
        self.frame_ids = [p.stem for p in self.rgb_files]
        if not (self.frame_ids == [p.stem for p in self.depth_files]
                == [p.stem for p in self.instance_files]
                == [p.name.removesuffix('_label.pkl') for p in self.label_files]):
            raise ValueError("HouseCat6D RGB/depth/instance/label frame IDs do not match")
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError('depth_scale must be positive and finite')
        self.K = np.loadtxt(self.root / "intrinsics.txt", dtype=np.float32).reshape(3, 3)
        if not self.rgb_files:
            raise ValueError(f"No frames found under {self.root}")

    def __len__(self):
        return len(self.rgb_files)

    def frame(self, index):
        rgb = self._read_image(self.rgb_files[index], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = self._read_image(self.depth_files[index], cv2.IMREAD_UNCHANGED).astype(np.float32) * self.depth_scale
        depth[depth <= 0] = 0
        instance = self._read_image(self.instance_files[index], cv2.IMREAD_UNCHANGED)
        if depth.ndim != 2 or instance.ndim != 2 or rgb.shape[:2] != depth.shape or depth.shape != instance.shape:
            raise ValueError(f'Inconsistent RGB/depth/instance image shapes at frame {self.frame_ids[index]}')
        with self.label_files[index].open("rb") as stream:
            labels = pickle.load(stream)
        return rgb, depth, instance, labels

    @staticmethod
    def _read_image(path, flags):
        image = cv2.imread(str(path), flags)
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        return image

    def mesh_path(self, model_name):
        path = self.mesh_dir / f"{model_name}.obj"
        if not path.is_file():
            raise FileNotFoundError(f"Missing CAD model for {model_name}: {path}")
        return path

    @staticmethod
    def object_records(labels):
        count = len(labels['model_list'])
        if not all(len(labels[key]) == count for key in ('instance_ids', 'class_ids', 'rotations', 'translations')):
            raise ValueError('Inconsistent object counts in labels')
        if len(set(labels['instance_ids'])) != count:
            raise ValueError('Duplicate instance IDs in labels')
        for index, (name, instance_id) in enumerate(zip(labels["model_list"], labels["instance_ids"])):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = np.asarray(labels["rotations"][index], dtype=np.float32)
            pose[:3, 3] = np.asarray(labels["translations"][index], dtype=np.float32)
            # Official vis_obj.py uses these rigid transforms directly with the
            # metric OBJ meshes. labels['scales'] must NOT be applied again.
            yield {"index": index, "model_name": name, "instance_id": int(instance_id),
                   "class_id": int(labels['class_ids'][index]), "pose_gt": pose}

    @staticmethod
    def mask_for_instance(instance_image, instance_id):
        return instance_image == int(instance_id)


def pose_errors(prediction, target, vertices):
    """Raw, symmetry-unaware pose errors in metres and degrees."""
    pred = vertices @ prediction[:3, :3].T + prediction[:3, 3]
    gt = vertices @ target[:3, :3].T + target[:3, 3]
    cosine = (np.trace(prediction[:3, :3] @ target[:3, :3].T) - 1) / 2
    return {
        'add_error_m': float(np.linalg.norm(pred - gt, axis=1).mean()),
        'translation_error_m': float(np.linalg.norm(prediction[:3, 3] - target[:3, 3])),
        'rotation_error_deg': float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))),
    }
