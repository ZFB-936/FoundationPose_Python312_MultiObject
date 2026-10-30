# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import kornia
import torch
from Utils import depth2xyzmap_batch
from learning.pose_data import BatchPoseData


class PoseRefinePreprocessor:
  def __init__(self, cfg):
    self.cfg = cfg

  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
      depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapAs[:,2:3]<0.001
    batch.xyz_mapAs = batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

    if batch.xyz_mapBs is None:
      depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapBs = kornia.geometry.transform.warp_perspective(batch.xyz_mapBs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapBs[:,2:3]<0.001
    batch.xyz_mapBs = batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

    return batch

  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    '''Transform the batch before feeding to the network
    !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
    '''
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch

class PoseScorePreprocessor(PoseRefinePreprocessor):
  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
      depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    invalid = batch.xyz_mapAs[:,2:3]<0.1
    batch.xyz_mapAs = (batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

    if batch.xyz_mapBs is None:
      depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapBs = kornia.geometry.transform.warp_perspective(batch.xyz_mapBs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    invalid = batch.xyz_mapBs[:,2:3]<0.1
    batch.xyz_mapBs = (batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

    return batch

  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch
