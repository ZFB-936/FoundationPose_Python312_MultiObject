# HouseCat6D 多目标 6D 位姿估计

入口 `run_housecat6d.py` 使用 HouseCat6D 自带的可见实例分割和固定实例 ID，跳过检测器和实例关联。它评估的是已知 CAD、已知实例身份的位姿估计与跟踪，不评估目标检测能力。

## 已复制的数据和资产

本地场景：`demo_data/scene01`，291 帧、852×1096 图像、5 个物体。

| 实例 ID | CAD 名称 |
| --- | --- |
| 1 | box-kleenex |
| 2 | can-kidney_beans |
| 3 | cup-green_actys |
| 4 | cup-new_york_big |
| 5 | remote-grey |

上述五个物体的 `.obj`、`.mtl`、纹理 PNG 和 `_norm.txt` 需要从 HouseCat6D 的 `obj_models_small_size_final` 复制到 `demo_data/scene01/mesh/`，共 20 个文件约 93 MiB。它们是带纹理的视觉模型，不是 `collision/` 碰撞模型。CAD 文件和校验清单留在本地，不纳入 Git。

整个 `demo_data/` 沿用仓库的忽略规则，RGB-D 场景和 CAD 资产不会随 Git 提交或传输。在另一台机器运行时，需要单独准备 HouseCat6D 场景，并把 scene01 用到的五组 CAD 资产复制到上述路径。

```text
scene01/
  intrinsics.txt
  rgb/000000.png
  depth/000000.png
  instance/000000.png
  labels/000000_label.pkl
  mesh/<model_name>.obj
  mesh/<model_name>.mtl
  mesh/<model_name>_0.png
  ...
```

读取器按文件名检查帧一一对应，而不是只比较数量。`instance/000000_<model>.png` 是单物体完整投影，可能包含遮挡区域；本入口使用 `instance/000000.png` 的可见实例图，背景为 255。

## 坐标和单位

- 深度 PNG 按 `0.001` 转成米，可通过 `--depth_scale` 调整。
- 网格已经是米制。五个网格的范围与标签 `gt_scales` 一致，不能再乘标签 `scales`。
- `labels` 的 `rotations` 和 `translations` 组成原始 OBJ 坐标到相机坐标的刚体变换。相机采用正 Z 向前的约定，与 FoundationPose 接口一致。
- 坐标依据：[HouseCat6D 官方可视化实现](https://github.com/Junggy/HouseCat6D/blob/main/visualization/vis_obj.py)。已抽取首帧网格渲染验证轮廓对齐。
- `obj_pose_final/` 的物体世界位姿和 `camera_pose/` 不参与本入口的位姿预测。
- PKL 标签仅从可信的本地数据集读取。

## 运行

```bash
conda activate foundationpose312
export PYTHONNOUSERSITE=1

# 先运行三帧验证首帧初始化与后续跟踪。
python run_housecat6d.py --max_frames 3 --save_vis

# 实时窗口显示；按 q 退出
python run_housecat6d.py --realtime

# 全场景：五个物体首次可见时分别初始化，后续各自跟踪。
python run_housecat6d.py --save_vis

# 每帧独立估计，无前帧位姿输入，耗时明显更长。
python run_housecat6d.py --mode register --max_frames 3 --debug_dir debug/scene01_register
```

每个实例独立保存 `pose_last` 和 CAD 状态，五个实例共享评分网络、精修网络和渲染上下文；推理按实例串行执行。真值位姿只用于误差计算，不用于初始化或跟踪。没有足够可见有效深度时跳过该实例，再出现时重新初始化；它不是遮挡恢复算法，也没有加入漂移检测。如果实例 ID 对应的 CAD 名称发生变化，入口会报错。

可选参数：`--scene_dir`、`--mesh_dir`、`--max_frames`、`--mode track|register`、`--est_refine_iter`、`--track_refine_iter`、`--depth_scale`、`--debug_dir`、`--save_vis`、`--realtime`、`--verbose`。`--realtime` 需要桌面显示环境；按 `q` 退出窗口。

## 输出

默认目录 `debug/scene01_multi/`：

- `ob_in_cam/<instance_id>/<frame_id>.txt`：预测的原始网格到相机 4×4 位姿，平移单位为米。
- `predictions.jsonl`：逐帧逐物体的预测、状态、计时和误差，逐帧写入。
- `metrics.csv`：每个实例的耗时、ADD、平移误差和旋转误差。
- `summary.json`：整体及各物体统计、网络准备耗时和每物体准备耗时。
- `vis/<frame_id>.png`：加 `--save_vis` 时输出预测框、坐标轴和实例名称。

计时前后同步 CUDA；不包含模型/网格准备、图像读取、评估、保存和绘图。`first_frame_inference_ms` 是五个实例首帧推理之和，后续统计也是每帧所有实例串行推理的总和，不能当作单物体耗时。

当前指标是未经对称性处理的原始 ADD、旋转角度误差和原点平移误差，ADD 对原网格全部顶点取均值。这些是调试指标，不等同于 HouseCat6D 官方评测协议；对称物体可能具有外观等价的多个姿态。

## 本次全场景验证

在 RTX 5060 Ti、Python 3.12 环境中，以默认首帧 5 次精修、跟踪 2 次精修完成 291 帧、5 个实例，共 1455 个有效预测；同时输出 291 张可视化。五物体首帧推理总计约 36.19 秒，后续每帧五物体串行推理平均约 135.28 毫秒；不包含网络及网格准备。当前完整网格每个约 10 万三角形，不能套用轻量单物体 demo 的耗时。

全序列原始 ADD 平均 11.73 毫米，原点平移误差平均 11.67 毫米，未经对称处理的旋转误差平均 2.44 度。这是一次运行的调试结果，不代表官方评测分数。完整数值以本地 `debug/scene01_multi/summary.json` 为准。
