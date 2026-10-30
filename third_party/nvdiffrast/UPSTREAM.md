# Vendored nvdiffrast

- Upstream: https://github.com/NVlabs/nvdiffrast
- Commit: `253ac4fcea7de5f396371124af597e6cc957bfae`
- Version: `0.4.0`
- License: [NVIDIA Source Code License](LICENSE.txt)

The Python package, CUDA/C++ sources, packaging files, upstream README and
license are copied unchanged from this commit. Upstream documentation, samples,
Dockerfile and generated build artifacts are omitted. This is the complete
runtime implementation, including upstream backward operations; kernels have
not been pruned or rewritten.

Run `bash build_all.sh` from the FoundationPose repository to compile and install
this local copy into the active Python environment. The build runs in a temporary
directory to avoid reusing binaries from another Python, PyTorch or CUDA version.
