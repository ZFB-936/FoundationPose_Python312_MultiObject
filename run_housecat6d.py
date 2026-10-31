"""Multi-object HouseCat6D pose estimation with dataset-provided instance masks."""

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import trimesh

from Utils import (dr, draw_posed_3d_box, draw_xyz_axis, prepare_rgbd_frame,
                   set_logging_format, set_seed)
from estimater import FoundationPose
from housecat6d import HouseCat6DScene, pose_errors
from learning.inference.predict_pose_refine import PoseRefinePredictor
from learning.inference.predict_score import ScorePredictor


def timed_call(function, *args, **kwargs):
    torch.cuda.synchronize()
    start = perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    return result, (perf_counter() - start) * 1000


def make_state(record, scene, output_dir, scorer, refiner, glctx):
    mesh = trimesh.load(scene.mesh_path(record['model_name']), process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Invalid triangle mesh: {record['model_name']}")
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
        mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
        debug=0, debug_dir=(str(output_dir / f"instance_{record['instance_id']}")
                            if output_dir is not None else None),
    )
    return dict(model_name=record['model_name'], mesh=mesh, estimator=estimator,
                needs_register=True, last_frame=-1)


def summarize(rows):
    valid = [row for row in rows if row['pose'] is not None]
    result = {'predictions': len(valid), 'skipped': len(rows) - len(valid)}
    for key in ('add_error_m', 'translation_error_m', 'rotation_error_deg'):
        values = [row[key] for row in valid]
        result[key + '_mean'] = float(np.mean(values)) if values else None
    for stage in ('register', 'track'):
        values = [row['elapsed_ms'] for row in valid if row['stage'] == stage]
        result[stage + '_count'] = len(values)
        result[stage + '_mean_ms'] = float(np.mean(values)) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene_dir', default='demo_data/scene01')
    parser.add_argument('--mesh_dir', default=None, help='Default: scene_dir/mesh')
    parser.add_argument('--max_frames', type=int, default=0, help='0: all frames')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--depth_scale', type=float, default=0.001, help='PNG depth to metres')
    parser.add_argument('--mode', choices=('track', 'register'), default='track',
                        help='track: initialize then track; register: independent estimation every frame')
    parser.add_argument('--debug_dir', default='debug/scene01_multi')
    parser.add_argument('--save_vis', action='store_true', help='Save predicted 3D boxes and axes as PNGs')
    parser.add_argument('--save_results', action='store_true', help='Save JSON/CSV/pose files; disabled by default')
    parser.add_argument('--realtime', action='store_true', help='Show each annotated frame in an OpenCV window')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.max_frames < 0 or min(args.est_refine_iter, args.track_refine_iter) < 1:
        parser.error('Frame count must be nonnegative and iteration counts must be positive')

    set_logging_format(logging.INFO if args.verbose else logging.WARNING)
    set_seed(0)
    scene = HouseCat6DScene(args.scene_dir, args.mesh_dir, args.depth_scale)
    frame_count = min(args.max_frames, len(scene)) if args.max_frames else len(scene)
    save_outputs = args.save_results or args.save_vis
    output_dir = Path(args.debug_dir) if save_outputs or args.realtime else None
    if save_outputs or args.realtime:
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        (output_dir / 'vis').mkdir(exist_ok=True)

    def setup_networks():
        return ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()

    (scorer, refiner, glctx), network_setup_ms = timed_call(setup_networks)
    # Share network weights and rasterizer, but NEVER share pose_last across instances.
    objects, predictions, setup_records, frame_timings = {}, [], [], []
    processed_frames = 0
    print(f'Networks and renderer ready: {network_setup_ms:.1f} ms', flush=True)
    stream = (output_dir / 'predictions.jsonl').open('w') if args.save_results else None
    for frame_index in range(frame_count):
            rgb, depth, instance_image, labels = scene.frame(frame_index)
            depth_filtered, depth_tensor, xyz_map = prepare_rgbd_frame(depth, scene.K)
            rgb_tensor = torch.as_tensor(rgb, device='cuda')
            frame_id = scene.frame_ids[frame_index]
            vis = rgb.copy() if args.save_vis or args.realtime else None
            frame_rows = []
            for record in scene.object_records(labels):
                iid = record['instance_id']
                mask = scene.mask_for_instance(instance_image, iid)
                row = dict(frame=frame_index, frame_id=frame_id, instance_id=iid,
                           model_name=record['model_name'], class_id=record['class_id'],
                           visible_pixels=int(mask.sum()), stage='skipped', elapsed_ms=None,
                           pose=None, add_error_m=None, translation_error_m=None,
                           rotation_error_deg=None, reason=None)
                # Skip invisible/invalid observations, and reinitialize on reappearance.
                if np.count_nonzero(mask & (depth_filtered >= 0.001)) < 4:
                    if iid in objects:
                        objects[iid]['needs_register'] = True
                    row['reason'] = 'insufficient_visible_depth'
                else:
                    if iid not in objects:
                        state, setup_ms = timed_call(make_state, record, scene, output_dir, scorer, refiner, glctx)
                        objects[iid] = state
                        setup_records.append(dict(instance_id=iid, model_name=record['model_name'], elapsed_ms=setup_ms))
                    state = objects[iid]
                    if state['model_name'] != record['model_name']:
                        raise ValueError(f'Instance {iid} changed CAD identity at frame {frame_id}')
                    estimator = state['estimator']
                    if args.mode == 'register' or state['needs_register'] or state['last_frame'] != frame_index - 1:
                        # GT transforms are used only below for evaluation, never initialization.
                        estimator.pose_last = None
                        pose, elapsed = timed_call(estimator.register, scene.K, rgb_tensor, depth_tensor, mask,
                                                   iteration=args.est_refine_iter, depth_processed=True,
                                                   xyz_map=xyz_map, return_tensor=True)
                        stage = 'register'
                    else:
                        pose, elapsed = timed_call(estimator.track_one, rgb_tensor, depth_tensor, scene.K,
                                                   iteration=args.track_refine_iter, depth_processed=True,
                                                   xyz_map=xyz_map, return_tensor=True)
                        stage = 'track'
                    pose_np = pose.detach().cpu().numpy()
                    if not np.isfinite(pose_np).all() or estimator.pose_last is None:
                        state['needs_register'] = True
                        row.update(reason='invalid_pose', elapsed_ms=elapsed)
                    else:
                        state.update(needs_register=False, last_frame=frame_index)
                        row.update(stage=stage, elapsed_ms=elapsed, pose=pose_np.tolist(),
                                   **pose_errors(pose_np, record['pose_gt'], state['mesh'].vertices))
                        if args.save_results:
                            pose_dir = output_dir / 'ob_in_cam' / str(iid)
                            pose_dir.mkdir(parents=True, exist_ok=True)
                            np.savetxt(pose_dir / f'{frame_id}.txt', pose_np)
                        if vis is not None:
                            color = tuple(int(c) for c in ((iid * 73) % 200 + 55, (iid * 127) % 200 + 55, (iid * 179) % 200 + 55))
                            vis = draw_posed_3d_box(scene.K, vis, pose_np, state['mesh'].bounds,
                                                  line_color=color)
                            vis = draw_xyz_axis(vis, pose_np, scale=0.04, K=scene.K, is_input_rgb=True)
                            center = scene.K @ pose_np[:3, 3]
                            uv = tuple(np.clip(center[:2] / max(center[2], 1e-6), -10000, 10000).astype(int))
                            cv2.putText(vis, f'{iid}: {record["model_name"]}', uv,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                frame_rows.append(row)
                if stream is not None:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            if stream is not None:
                stream.flush()
            predictions.extend(frame_rows)
            frame_ms = sum(row['elapsed_ms'] or 0 for row in frame_rows)
            frame_timings.append(frame_ms)
            processed_frames += 1
            print(f'Frame {frame_id}: {sum(r["pose"] is not None for r in frame_rows)} objects, total inference {frame_ms:.1f} ms', flush=True)
            if vis is not None:
                display_frame = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                if args.save_vis:
                    cv2.imwrite(str(output_dir / 'vis' / f'{frame_id}.png'), display_frame)
                if args.realtime:
                    cv2.imshow('HouseCat6D multi-object 6D pose', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    summary = dict(scene_dir=str(scene.root.resolve()), frames=processed_frames,
                   requested_frames=frame_count, objects=len(objects),
                   mode=args.mode, network_setup_ms=network_setup_ms, object_setup=setup_records,
                   image_size=list(rgb.shape[:2]), gpu=torch.cuda.get_device_name(),
                   est_refine_iter=args.est_refine_iter, track_refine_iter=args.track_refine_iter,
                   depth_scale=args.depth_scale,
                   first_frame_inference_ms=frame_timings[0],
                   following_frame_inference_mean_ms=float(np.mean(frame_timings[1:])) if len(frame_timings) > 1 else None,
                   overall=summarize(predictions),
                   per_object={str(iid): dict(model_name=state['model_name'], **summarize([r for r in predictions if r['instance_id'] == iid])) for iid, state in objects.items()},
                   notes=['GT masks and IDs bypass detection and association; GT poses are evaluation-only.',
                          'Metrics are raw ADD/rotation/translation, not the official symmetry-aware HouseCat6D protocol.',
                          'Synchronized inference timings exclude setup, data loading, evaluation, saving and visualization.'])
    if stream is not None:
        stream.close()
    if args.save_results:
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        fields = [k for k in predictions[0] if k != 'pose']
        with (output_dir / 'metrics.csv').open('w', newline='') as stream_out:
            writer = csv.DictWriter(stream_out, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(predictions)
        print(f'Results saved to {output_dir.resolve()}', flush=True)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    if args.realtime:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
