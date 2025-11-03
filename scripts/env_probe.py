import importlib
import platform
import shutil
import sys

import torch


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return "not installed"

    version = getattr(module, "__version__", None)
    if version is None and name == "cv2":
        version = getattr(module, "__version__", None)
    if version is None:
        version = "unknown"
    return str(version)


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    mps_built = bool(mps_backend and mps_backend.is_built())
    print(f"MPS available: {mps_available} (built: {mps_built})")
    print(f"CUDA available: {torch.cuda.is_available()}")

    packages = ["ultralytics", "cv2", "faiss", "timm", "albumentations"]
    for pkg in packages:
        print(f"{pkg}: {module_version(pkg)}")

    print(f"sys.executable: {sys.executable}")
    pip_path = shutil.which("pip")
    print(f"pip: {pip_path if pip_path else 'not found'}")


if __name__ == "__main__":
    main()
