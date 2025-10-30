set shell := ["bash", "-cu"]

# 1️⃣ Install all Poetry dependencies
install-poetry:
    poetry install

# 2️⃣ Upgrade pip, setuptools, and wheel
install-setuptools:
    poetry run pip install --upgrade pip setuptools wheel

# 3️⃣ Install detectron2 on Windows
install-detectron2-windows:
    poetry run pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# 4️⃣ Install detectron2 on macOS M1 (Apple Silicon)
install-detectron2-macos-m1:
    CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" \
    poetry run pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# 5️⃣ Aliases for full installation
# Windows installation: Poetry + setuptools upgrade + detectron2
install-win: install-poetry install-setuptools install-detectron2-windows

# macOS M1 installation: Poetry + setuptools upgrade + detectron2
install-m1: install-poetry install-setuptools install-detectron2-macos-m1
