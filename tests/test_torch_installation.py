import torch

def test_torch_installation():
    if torch.cuda.is_available():
        print("✅ CUDA is available.")
        assert torch.cuda.device_count() > 0, "No CUDA devices detected"
        assert torch.__version__.startswith("2."), f"Unexpected torch version: {torch.__version__}"
    else:
        print("⚠️ CUDA is not available. Running in CPU-only mode.")
