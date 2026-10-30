# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from estimater import FoundationPose
from datareader import YcbineoatReader
from learning.inference.predict_score import ScorePredictor
from learning.inference.predict_pose_refine import PoseRefinePredictor
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter


def timed_call(function, *args, **kwargs):
    """Include CPU work, transfers and GPU execution; exclude earlier GPU work."""
    torch.cuda.synchronize()
    start = perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    return result, (perf_counter() - start) * 1000


def save_timings(output_dir, records, setup_ms, metadata, skip_tracking=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'timing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame_index', 'frame_id', 'stage', 'elapsed_ms'])
        writer.writeheader()
        writer.writerows(records)

    tracking = [row['elapsed_ms'] for row in records if row['stage'] == 'track']
    measured = tracking[skip_tracking:]
    summary = {
        'metadata': metadata,
        'scope': 'CUDA-synchronized wall time; excludes image/mask reading, pose saving and display',
        'model_setup_ms': setup_ms,
        'first_frame_register_ms': records[0]['elapsed_ms'] if records else None,
        'tracking_frames_total': len(tracking),
        'tracking_frames_skipped': min(skip_tracking, len(tracking)),
        'tracking_frames_measured': len(measured),
        'first_tracking_frame_ms': tracking[0] if tracking else None,
        'tracking_mean_ms': float(np.mean(measured)) if measured else None,
        'tracking_median_ms': float(np.median(measured)) if measured else None,
        'tracking_p95_ms': float(np.percentile(measured, 95)) if measured else None,
        'tracking_min_ms': min(measured) if measured else None,
        'tracking_max_ms': max(measured) if measured else None,
        'tracking_fps': 1000 / float(np.mean(measured)) if measured and sum(measured) > 0 else None,
    }
    with (output_dir / 'timing_summary.json').open('w') as stream:
        json.dump(summary, stream, indent=2)
    return summary


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/custom_dataset/cup/mesh/untitled.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/custom_dataset/cup')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--benchmark', action='store_true', help='Disable debug rendering and verbose logs for timing')
  parser.add_argument('--max_frames', type=int, default=0, help='Maximum frames including initialization; 0 uses all frames')
  parser.add_argument('--skip_tracking', type=int, default=0, help='Exclude this many initial tracking frames from aggregate statistics')
  args = parser.parse_args()
  if args.max_frames < 0 or args.skip_tracking < 0:
    parser.error('--max_frames and --skip_tracking must be nonnegative')

  set_logging_format(level=logging.WARNING if args.benchmark else logging.INFO)
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = 0 if args.benchmark else args.debug
  debug_dir = args.debug_dir
  os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  def prepare_estimator():
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    return FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)

  est, setup_ms = timed_call(prepare_estimator)
  print(f'Model setup: {setup_ms:.2f} ms', flush=True)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  records = []
  frame_count = min(args.max_frames, len(reader)) if args.max_frames else len(reader)
  for i in range(frame_count):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose, elapsed_ms = timed_call(est.register, K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      stage = 'register'

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose, elapsed_ms = timed_call(est.track_one, rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
      stage = 'track'

    records.append(dict(frame_index=i, frame_id=reader.id_strs[i], stage=stage, elapsed_ms=elapsed_ms))
    print(f'Frame {i:04d} {stage}: {elapsed_ms:.2f} ms', flush=True)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

  summary = save_timings(debug_dir, records, setup_ms, metadata={
    'gpu': torch.cuda.get_device_name(), 'torch_version': str(torch.__version__),
    'mesh_file': args.mesh_file, 'test_scene_dir': args.test_scene_dir,
    'image_size': [reader.H, reader.W], 'est_refine_iter': args.est_refine_iter,
    'track_refine_iter': args.track_refine_iter, 'debug': debug,
    'benchmark': args.benchmark,
  }, skip_tracking=args.skip_tracking)
  print(f"First-frame initialization: {summary['first_frame_register_ms']:.2f} ms")
  if summary['tracking_mean_ms'] is not None:
    print(f"Tracking ({summary['tracking_frames_measured']} frames): "
          f"mean={summary['tracking_mean_ms']:.2f} ms, "
          f"median={summary['tracking_median_ms']:.2f} ms, "
          f"P95={summary['tracking_p95_ms']:.2f} ms, "
          f"FPS={summary['tracking_fps']:.2f}")
  else:
    print('No tracking frames available for aggregate statistics.')
  print(f'Timing reports: {debug_dir}/timing.csv and timing_summary.json')

