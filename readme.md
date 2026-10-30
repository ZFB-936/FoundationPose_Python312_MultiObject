# FoundationPose：基于 CAD 模型的 6D 位姿估计与跟踪

本仓库保留已知物体 CAD 模型的位姿估计与跟踪流程。输入为物体网格、RGB-D 图像、相机内参和首帧物体掩码；首帧估计位姿，后续帧进行跟踪。

原项目：[FoundationPose](https://github.com/NVlabs/FoundationPose) · [论文](https://arxiv.org/abs/2312.08344)

## 代码结构

```text
run_demo.py                         # 单物体估计与视频跟踪
run_linemod.py                      # LINEMOD 数据集入口
run_ycb_video.py                    # YCB-Video 数据集入口
estimater.py                       # 候选位姿生成、精修、评分与跟踪
Utils.py                           # CAD 渲染、深度处理、几何和可视化
datareader.py                     # RGB-D、掩码、内参与 CAD 模型读取
learning/
  rotation.py                      # 纯 PyTorch 旋转转换
  models/                          # 精修网络、评分网络及共享模块
  inference/predict_pose_refine.py   # 精修推理
  inference/predict_score.py         # 评分推理
  preprocessing.py                 # RGB-D 裁剪后的输入预处理
  pose_data.py                     # 批量位姿数据结构
mycpp/                             # 候选位姿聚类扩展
weights/                           # 预训练权重
```

两个预测器直接使用独立的输入预处理模块，不再依赖 HDF5 数据集类。权重配置使用 YAML。

## 环境安装

以下配置沿用本仓库的 Python 3.9、PyTorch 2.7.1 和 CUDA 12.8 版本。安装前准备可用的 NVIDIA 驱动和 CUDA 开发工具链。

```bash
conda create -n foundationpose python=3.9 -y
conda activate foundationpose
conda install -c conda-forge eigen=3.4.0 cmake boost-cpp -y
python -m pip install -r requirements.txt

# CAD 渲染器仍为必需依赖。
python -m pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# 仅编译候选位姿聚类扩展。
bash build_all_conda.sh -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
```

旋转转换已改为纯 PyTorch 实现，无需额外的三维深度学习工具库。依赖清单只保留当前代码使用的包，不包含 Notebook、纹理展开、资产下载、检测器或实验管理工具。

迁移仓库路径后，如果 CMake 报旧路径缓存错误，可先移走 `mycpp/build/`，再重新运行构建命令。

## Docker

在仓库根目录构建，镜像使用同一份依赖清单：

```bash
docker build -f docker/dockerfile -t foundationpose .
cd docker
bash run_container.sh
```

首次进入容器后，在挂载的仓库中编译扩展：

```bash
bash build_all.sh -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
```

## 数据和权重

下载[预训练权重](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing)，目录如下：

```text
weights/2023-10-28-18-33-37/{config.yml,model_best.pth}  # 精修
weights/2024-01-11-20-02-45/{config.yml,model_best.pth}  # 评分
```

[演示数据](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing)解压到 `demo_data/`，也可以通过命令行指定自己的 CAD 网格和 RGB-D 序列。网格和深度均使用米作为单位。

## 运行

```bash
python run_demo.py --mesh_file /path/to/model.obj --test_scene_dir /path/to/rgbd_sequence
python run_linemod.py --linemod_dir /path/to/linemod/000005
python run_ycb_video.py --ycbv_dir /path/to/YCB_Video
```

`run_demo.py` 保留本地自定义数据集的默认路径。运行时按需指定 `--debug_dir`；位姿输出在该目录的 `ob_in_cam/` 中。

## 检查

```bash
python -m unittest discover -s tests -v
python run_demo.py --help
python run_linemod.py --help
python run_ycb_video.py --help
```

## 引用与许可

```bibtex
@InProceedings{foundationposewen2024,
  author = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
  title = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
  booktitle = {CVPR},
  year = {2024}
}
```

代码及数据遵循 [NVIDIA Source Code License](LICENSE)。原作者及版权声明保留在源文件中。
