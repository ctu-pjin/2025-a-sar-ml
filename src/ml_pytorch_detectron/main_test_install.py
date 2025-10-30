import sys
import torch

def test_torch():
    try:
        print(f"✅ PyTorch version: {torch.__version__}")

        # Check CUDA (NVIDIA GPU) - won't be available on M1
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Check MPS (Apple Metal)
        mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        print(f"MPS (Apple GPU) available: {mps_available}")

        # Print device info
        if mps_available:
            device = torch.device("mps")
            print(f"Using MPS device: {device}")
        else:
            device = torch.device("cpu")
            print("Using CPU (no GPU available)")

    except ImportError:
        print("❌ PyTorch is not installed")
        sys.exit(1)

def test_detectron2():
    try:
        import detectron2
        from detectron2.utils.logger import setup_logger
        setup_logger()
        print(f"✅ Detectron2 version: {detectron2.__version__}")
    except ImportError:
        print("❌ Detectron2 is not installed")
        sys.exit(1)

if __name__ == "__main__":
    print("Testing PyTorch installation...")
    test_torch()
    print("\nTesting Detectron2 installation...")
    test_detectron2()
    print("\n🎉 All checks passed!")
