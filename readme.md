# FoundationPose Python 3.12 Multi-Object

<p align="center">
  <a href="#english">English</a> | <a href="#中文说明">中文</a> | <a href="#视频演示">视频演示 / Videos</a>
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch 2.11" src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="CUDA 12.8" src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white">
  <img alt="FoundationPose" src="https://img.shields.io/badge/FoundationPose-CAD--based-2E7D32">
  <img alt="HouseCat6D" src="https://img.shields.io/badge/Dataset-HouseCat6D-6A5ACD">
  <img alt="NVIDIA Source Code License" src="https://img.shields.io/badge/License-NVIDIA%20Source%20Code-555555">
</p>

## English

This fork keeps the CAD-based FoundationPose pipeline and adds an offline multi-object 6D pose evaluation path for HouseCat6D.

The fork has two major changes:

1. **A simplified Python 3.12 runtime.** The model-free BundleSDF path and its Kaolin/mycuda dependencies were removed. The repository uses PyTorch 2.11 with CUDA 12.8, pure-PyTorch rotation conversions, and a vendored nvdiffrast build. The original refinement and scoring network weights are unchanged.
2. **HouseCat6D multi-object pose estimation.** The new `run_housecat6d.py` reads HouseCat6D's per-frame instance masks and stable instance IDs, runs an independent FoundationPose state for each CAD object, and reports pose and timing metrics. It does not require YOLO or an instance-ID tracker for this dataset evaluation.

The upstream project is [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose), presented at CVPR 2024. This fork is intended for known CAD models and RGB-D input; it does not solve object detection by itself.

## Video Demos

| Multi-object 6D pose estimation | Single-object 6D pose estimation on Ubuntu 24 / RTX 5060 Ti |
| --- | --- |
| [![Multi-object 6D pose estimation](https://i1.hdslb.com/bfs/archive/435568b31b15dbda2cc3802abb770ebd4a8fa7e3.jpg)](https://www.bilibili.com/video/BV1Kbx163EJi/) | [![Single-object FoundationPose on Ubuntu 24 and RTX 5060 Ti](https://i1.hdslb.com/bfs/archive/c45fb0965d825da2bdcafe63a42dd8092d7150af.jpg)](https://www.bilibili.com/video/BV1VVx16HEea/) |
| [Watch on Bilibili](https://www.bilibili.com/video/BV1Kbx163EJi/) | [Watch on Bilibili](https://www.bilibili.com/video/BV1VVx16HEea/) |

## Changes from Upstream

| Area | Upstream FoundationPose | This fork |
| --- | --- | --- |
| Python/runtime | Older Conda/PyTorch combinations documented | Python 3.12, PyTorch 2.11, CUDA 12.8 |
| Operating modes | Model-based and model-free BundleSDF paths | CAD-based model path only |
| Model-free dependencies | BundleSDF, Kaolin, and `bundlesdf/mycuda` | Removed |
| Rotation dependency | PyTorch3D transforms | Pure PyTorch conversions in `learning/rotation.py` |
| NVDiffRast | Installed separately from GitHub | Pinned source vendored under `third_party/nvdiffrast/` and built locally |
| Data path | Single-object demos and benchmark readers | Adds HouseCat6D multi-object evaluation using dataset masks and IDs |
| Multi-object tracking | No dedicated multi-object entry point | Independent pose state per CAD instance, with optional real-time visualization |

The refinement and scoring network architectures, checkpoint names, and CAD render-and-compare design are retained. This is an engineering fork, not a newly trained model release.

## Performance Reference

Reported tracking benchmark for a 736-frame sequence:

```text
Tracking (736 frames): mean=14.64 ms, median=13.70 ms,
P95=19.51 ms, FPS=68.31
```

This is a reference for this fork, not a controlled upstream comparison. Reproduce it with the same GPU, Python/PyTorch/CUDA versions, CAD assets, image resolution, refinement iterations, and measurement scope before comparing against another checkout. The `run_demo.py --benchmark` timing scope excludes image loading, result writing, and display.

## Features

- Python 3.12 and CUDA 12.8 environment.
- Model-based RGB-D 6D pose initialization and tracking.
- Vendored nvdiffrast source compiled by the local build script.
- Per-frame shared depth filtering and XYZ computation for multi-object inference.
- GPU-resident pose tensors through refinement and scoring.
- Optional real-time visualization with OpenCV.
- HouseCat6D evaluation with ADD, translation, rotation, and timing reports.

## Repository Layout

```text
run_demo.py                    Single-object demo and timing benchmark
run_housecat6d.py              HouseCat6D multi-object evaluation
housecat6d.py                  HouseCat6D reader and pose metrics
estimater.py                   Candidate generation, refinement, scoring, tracking
Utils.py                       CAD rendering, depth processing, geometry
learning/models/               Refinement and scoring networks
learning/inference/            Refinement and scoring inference
third_party/nvdiffrast/        Vendored GPU renderer source and license
mycpp/                         Initial-pose clustering extension
docs/housecat6d.md             Dataset preparation and evaluation details
tests/                         Unit and regression tests
```

## Environment

The tested environment is Python 3.12, PyTorch 2.11.0 / torchvision 0.26.0 with CUDA 12.8, Open3D 0.20, NumPy 2.5, Kornia 0.9, and Warp 1.18.

```bash
conda create -n foundationpose312 python=3.12 -y
conda activate foundationpose312
export PYTHONNOUSERSITE=1
conda install -c conda-forge eigen=3.4.0 cmake boost-cpp -y
python -m pip install setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt

# Compile the vendored nvdiffrast and mycpp extensions.
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=4 bash build_all.sh
```

If OpenCV prints `QFontDatabase: Cannot find font directory .../cv2/qt/fonts` when using `--realtime`, run `bash build_all.sh` once after installing system DejaVu fonts. The script links `/usr/share/fonts/truetype/dejavu` into the OpenCV Qt package. Headless inference does not need Qt or fonts.

`build_all.sh` does not download nvdiffrast from GitHub. It builds the pinned source under `third_party/nvdiffrast/` and then builds the pose-clustering extension. The nvdiffrast source is retained under its upstream license; see `third_party/nvdiffrast/UPSTREAM.md`.

For a different GPU, set `TORCH_CUDA_ARCH_LIST` to its CUDA compute capability. For example, the local RTX 5060 Ti uses `12.0`.

## Single-Object Demo

Download the FoundationPose weights and demo data from the links in the upstream project, then run:

```bash
python run_demo.py --mesh_file /path/to/model.obj --test_scene_dir /path/to/rgbd_sequence
```

The timing mode reports model setup, first-frame initialization, and subsequent tracking:

```bash
python run_demo.py --benchmark --max_frames 50
```

## HouseCat6D Multi-Object Evaluation

The HouseCat6D reader uses the dataset's `instance/*.png` masks and `labels/*_label.pkl` files. Each object gets its own pose state; detection and cross-frame association are bypassed because the dataset supplies stable IDs.

Prepare `scene01` and copy these five CAD models into `demo_data/scene01/mesh/`:

```text
box-kleenex
can-kidney_beans
cup-green_actys
cup-new_york_big
remote-grey
```

CAD and RGB-D assets are intentionally ignored by Git and are not included in this repository. Copy the metric textured `.obj`, `.mtl`, texture PNG, and `_norm.txt` files from HouseCat6D's `obj_models_small_size_final` directory. See [docs/housecat6d.md](docs/housecat6d.md) for the exact layout and coordinate convention.

```bash
# Short check
python run_housecat6d.py --max_frames 3

# Complete scene
python run_housecat6d.py

# Independent initialization on every frame
python run_housecat6d.py --mode register --max_frames 3
```

Results are not written by default. Enable them explicitly:

```bash
python run_housecat6d.py --save_results
python run_housecat6d.py --save_vis
python run_housecat6d.py --realtime    # press q to exit
```

The metrics are raw, symmetry-unaware ADD, translation, and rotation errors for debugging; they are not the official HouseCat6D benchmark protocol.

The multi-object path keeps the original weights, CAD meshes, candidate count, and refinement iterations. It shares frame-level RGB-D preprocessing, keeps pose tensors on the GPU, caches XYZ pixel grids, and avoids clearing the CUDA allocator between calls.

## Tests

```bash
python -m unittest discover -s tests -v
python run_demo.py --help
python run_housecat6d.py --help
```

## 中文说明

### 视频演示

| 多目标 6D 位姿估计 | Ubuntu 24 / RTX 5060 Ti 单目标 6D 位姿估计 |
| --- | --- |
| [![多目标 6D 位姿估计](https://i1.hdslb.com/bfs/archive/435568b31b15dbda2cc3802abb770ebd4a8fa7e3.jpg)](https://www.bilibili.com/video/BV1Kbx163EJi/) | [![Ubuntu 24 与 RTX 5060 Ti 上的单目标位姿估计](https://i1.hdslb.com/bfs/archive/c45fb0965d825da2bdcafe63a42dd8092d7150af.jpg)](https://www.bilibili.com/video/BV1VVx16HEea/) |
| [哔哩哔哩观看](https://www.bilibili.com/video/BV1Kbx163EJi/) | [哔哩哔哩观看](https://www.bilibili.com/video/BV1VVx16HEea/) |

### 项目定位

本分支保留基于 CAD 模型的 FoundationPose RGB-D 六维位姿估计与跟踪，并增加 HouseCat6D 多物体离线评估入口。

### 与原仓库的对比

| 方面 | 原始 FoundationPose | 当前分支 |
| --- | --- | --- |
| Python/运行环境 | 文档中使用较旧的 Conda/PyTorch 组合 | Python 3.12、PyTorch 2.11、CUDA 12.8 |
| 工作模式 | model-based + model-free BundleSDF | 只保留 CAD model-based |
| model-free 依赖 | BundleSDF、Kaolin、`bundlesdf/mycuda` | 已删除 |
| 旋转变换 | 依赖 PyTorch3D transforms | `learning/rotation.py` 中的纯 PyTorch 实现 |
| NVDiffRast | 从 GitHub 单独安装 | 固定源码内置在 `third_party/nvdiffrast/` 并统一编译 |
| 多物体入口 | 没有专用多物体入口 | 新增 HouseCat6D 多物体评估、独立实例状态和实时显示 |

精修器和评分器网络结构、预训练权重命名以及 CAD 渲染比较流程保持不变；当前分支是工程精简和多物体评估扩展，不是重新训练的新模型。

### 性能参考

一次 736 帧序列跟踪运行报告的结果：

```text
Tracking (736 frames): mean=14.64 ms, median=13.70 ms,
P95=19.51 ms, FPS=68.31
```

该结果是当前分支的参考值，不是与上游进行的受控对比。比较不同版本时，应保持 GPU、Python/PyTorch/CUDA、CAD 资产、图像分辨率、精修迭代次数和计时范围一致。`run_demo.py --benchmark` 的计时不包含图像读取、结果保存和显示。

### 两项主要修改

1. **Python 3.12 和精简架构**：删除 model-free BundleSDF、Kaolin 和 mycuda 路径；使用 Python 3.12、PyTorch 2.11/CUDA 12.8；旋转转换改为纯 PyTorch；内置 nvdiffrast 源码并由 `build_all.sh` 统一编译。
2. **HouseCat6D 多物体 6D 位姿**：使用数据集自带的实例 mask 和稳定实例 ID，为每个 CAD 物体维护独立的 FoundationPose 状态，支持初始化、跟踪、实时显示和误差统计，不需要 YOLO。

### 中文运行示例

```bash
conda activate foundationpose312
python run_housecat6d.py --max_frames 3
python run_housecat6d.py --save_results
python run_housecat6d.py --realtime
```

默认不保存 JSON、CSV、位姿或 PNG 文件；需要保存时显式添加 `--save_results` 或 `--save_vis`。HouseCat6D 的 RGB-D 数据和 CAD 资产不提交到 Git，需要根据 [中文数据说明](docs/housecat6d.md) 单独准备。

## License

The original FoundationPose files and vendored nvdiffrast source retain their respective copyright and license notices. See [LICENSE](LICENSE) and `third_party/nvdiffrast/LICENSE.txt`.
