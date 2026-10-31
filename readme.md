# FoundationPose Python 3.12 Multi-Object

This fork keeps the CAD-based FoundationPose pipeline and adds an offline multi-object 6D pose evaluation path for HouseCat6D.

The fork has two major changes:

1. **A simplified Python 3.12 runtime.** The model-free BundleSDF path and its Kaolin/mycuda dependencies were removed. The repository uses PyTorch 2.11 with CUDA 12.8, pure-PyTorch rotation conversions, and a vendored nvdiffrast build. The original refinement and scoring network weights are unchanged.
2. **HouseCat6D multi-object pose estimation.** The new `run_housecat6d.py` reads HouseCat6D's per-frame instance masks and stable instance IDs, runs an independent FoundationPose state for each CAD object, and reports pose and timing metrics. It does not require YOLO or an instance-ID tracker for this dataset evaluation.

The upstream project is [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose), presented at CVPR 2024. This fork is intended for known CAD models and RGB-D input; it does not solve object detection by itself.

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

### 项目定位

本分支保留基于 CAD 模型的 FoundationPose RGB-D 六维位姿估计与跟踪，并增加 HouseCat6D 多物体离线评估入口。

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
