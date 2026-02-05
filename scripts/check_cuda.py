"""Check CUDA availability for PyTorch."""

import torch

print("=" * 60)
print("CUDA Configuration Check")
print("=" * 60)
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\n✓ GPU está disponível e será usado pelo BERT!")
else:
    print(f"\n✗ GPU não disponível. BERT usará CPU (muito mais lento)")
    print(f"\nPara usar GPU:")
    print(f"  1. Instale PyTorch com CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print(f"  2. Verifique se tem GPU NVIDIA compatível")
