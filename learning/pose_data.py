# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class BatchPoseData:
    """
    rgbs: (bsz, 3, h, w) torch tensor uint8
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    K: (bsz, 3, 3) float32
    """

    rgbs: torch.Tensor = None
    object_datas = None
    bboxes: torch.Tensor = None
    K: torch.Tensor = None
    depths: Optional[torch.Tensor] = None
    rgbAs = None
    rgbBs = None
    depthAs = None
    depthBs = None
    normalAs = None
    normalBs = None
    poseA = None  #(B,4,4)
    poseB = None
    targets = None  # Score targets, torch tensor (B)

    def __init__(self, rgbAs=None, rgbBs=None, depthAs=None, depthBs=None, normalAs=None, normalBs=None, maskAs=None, maskBs=None, poseA=None, poseB=None, xyz_mapAs=None, xyz_mapBs=None, tf_to_crops=None, Ks=None, crop_masks=None, model_pts=None, mesh_diameters=None, labels=None):
        self.rgbAs = rgbAs
        self.rgbBs = rgbBs
        self.depthAs = depthAs
        self.depthBs = depthBs
        self.normalAs = normalAs
        self.normalBs = normalBs
        self.poseA = poseA
        self.poseB = poseB
        self.maskAs = maskAs
        self.maskBs = maskBs
        self.xyz_mapAs = xyz_mapAs
        self.xyz_mapBs = xyz_mapBs
        self.tf_to_crops = tf_to_crops
        self.crop_masks = crop_masks
        self.Ks = Ks
        self.model_pts = model_pts
        self.mesh_diameters = mesh_diameters
        self.labels = labels


    def pin_memory(self) -> "BatchPoseData":
        for k in self.__dict__:
            if self.__dict__[k] is not None:
              try:
                self.__dict__[k] = self.__dict__[k].pin_memory()
              except Exception as e:
                pass
        return self

    def cuda(self):
        for k in self.__dict__:
            if self.__dict__[k] is not None:
              try:
                self.__dict__[k] = self.__dict__[k].cuda()
              except:
                pass
        return self

    def select_by_indices(self, ids):
      out = BatchPoseData()
      for k in self.__dict__:
        if self.__dict__[k] is not None:
          out.__dict__[k] = self.__dict__[k][ids.to(self.__dict__[k].device)]
      return out
