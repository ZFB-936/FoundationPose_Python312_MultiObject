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
third_party/nvdiffrast/             # 内置 GPU 渲染库源码、原许可证和版本说明
build_all.sh                       # 统一编译并安装渲染库和聚类扩展
weights/                           # 预训练权重
```

两个预测器直接使用独立的输入预处理模块，不再依赖 HDF5 数据集类。权重配置使用 YAML。

## 环境安装

本分支使用 Python 3.12、PyTorch 2.11.0 / torchvision 0.26.0（CUDA 12.8），并更新 Open3D 0.20、NumPy 2.5、Kornia 0.9 和 Warp 1.18。运行依赖版本见 `requirements.txt`。安装前准备可用的 NVIDIA 驱动和 CUDA 12.8 开发工具链。

```bash
conda create -n foundationpose312 python=3.12 -y
conda activate foundationpose312
export PYTHONNOUSERSITE=1
conda install -c conda-forge eigen=3.4.0 cmake boost-cpp -y
python -m pip install setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
# 普通包使用清华镜像；CUDA 版 PyTorch 通过 requirements.txt 中的官方额外源获取。
python -m pip install -r requirements.txt
# 从仓库内置源码编译 nvdiffrast，并编译 mycpp。
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=4 bash build_all.sh
```

旋转转换已改为纯 PyTorch 实现，无需额外的三维深度学习工具库。依赖清单只保留当前代码使用的包，不包含 Notebook、纹理展开、资产下载、检测器或实验管理工具。

`mycpp` 按解释器分别构建到 `mycpp/build/cpython-312/` 等目录，不会覆盖旧环境的扩展。切换 Python 后需重新运行构建命令。nvdiffrast 0.4.0 已内置到 `third_party/nvdiffrast/`，来源提交为 `253ac4fcea7de5f396371124af597e6cc957bfae`，保留原始许可证与版权声明。运行时源码与上游保持一致，未裁剪 CUDA 内核；省略了示例、文档资源和生成文件。

`build_all.sh` 先编译并安装内置 nvdiffrast，再编译 `mycpp`。构建阶段不从 GitHub 下载源码，也不联网安装依赖，因此需先完成 `requirements.txt` 的安装，并准备 CMake、C++ 编译器和 CUDA Toolkit。渲染库在临时目录中构建，避免复用不同 Python/PyTorch/CUDA 环境的旧产物；重复执行会重新编译并替换当前环境中的 nvdiffrast。

`MAX_JOBS` 控制两个扩展的编译并发数，默认 4。默认检测本机 GPU 架构；在无 GPU 的构建环境中需显式设置 `TORCH_CUDA_ARCH_LIST`（例如 RTX 5060 Ti 使用 `12.0`）。`build_all_conda.sh` 保留为同一脚本的兼容入口，附加参数会传给 `mycpp` 的 CMake 配置。

AMP 调用已更新为 `torch.amp.autocast`，权重按 `weights_only=True` 加载，并移除了推理时修改全局默认张量类型的旧调用。

## Docker

在仓库根目录构建，镜像使用同一份依赖清单：

```bash
docker build -f docker/dockerfile --build-arg TORCH_CUDA_ARCH_LIST=12.0 -t foundationpose .
cd docker
bash run_container.sh
```

`12.0` 对应本地 RTX 5060 Ti；其他显卡需改为对应的计算能力。Docker 构建时也调用同一脚本编译两个扩展，镜像尚未实际构建验证。首次进入容器后，挂载目录会覆盖镜像中的构建目录，需要在挂载的仓库中执行：

```bash
bash build_all.sh
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

使用计时模式可以关闭显示与调试渲染，分别统计模型准备、首帧初始化和后续跟踪：

```bash
python run_demo.py --benchmark --debug_dir debug/python312
```

计时前后同步 CUDA，不包含图像读取、结果保存和显示；报告写入 `timing.csv` 和 `timing_summary.json`。首次调用仍可能包含编译、内核加载等开销。

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
