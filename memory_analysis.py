import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np

# Simplified memory usage checker
def check_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("CUDA not available")

def calculate_tensor_memory(shape):
    """Calculate memory usage for a tensor given its shape"""
    elements = 1
    for dim in shape:
        elements *= dim
    # 4 bytes per float32
    memory_mb = (elements * 4) / (1024 * 1024)
    return memory_mb

def main():
    print("=== Memory Usage Analysis for 128x128 Steganography Model ===\n")
    
    # Calculate expected memory for different tensors in the model
    print("Tensor Memory Calculations:")
    print("- Input tensor (batch=1, 6 channels, 128x128):", calculate_tensor_memory([1, 6, 128, 128]), "MB")
    print("- After ILWT/DWT (batch=1, 24 channels, 128x128):", calculate_tensor_memory([1, 24, 128, 128]), "MB")
    print("- Hidden layer (batch=1, 24 channels, 128x128):", calculate_tensor_memory([1, 24, 128, 128]), "MB")
    
    print("\nActual Memory Usage:")
    check_memory_usage()
    
    # Create a simple model for memory testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test basic operations memory impact
    print(f"\nMemory before tensor creation: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB" if torch.cuda.is_available() else "CUDA not available")
    
    if torch.cuda.is_available():
        # Create sample tensors to see memory impact
        x = torch.randn(1, 6, 128, 128, device=device)
        print(f"Memory after input tensor (1,6,128,128): {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        
        # Simulate the ILWT operation 
        conv = nn.Conv2d(6, 24, kernel_size=2, stride=2, groups=6, device=device)
        y = conv(x)
        print(f"Memory after ILWT-like operation: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        
        # Simulate hidden layer
        hidden_conv = nn.Conv2d(24, 24, kernel_size=3, padding=1, device=device)
        z = hidden_conv(y)
        print(f"Memory after hidden layer: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        
        # Clean up
        del x, y, z, conv, hidden_conv
        torch.cuda.empty_cache()
        print(f"Memory after cleanup: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    print("\n=== Memory Optimization Notes ===")
    print("The minimum VRAM needed for 128x128 images:")
    print("- Input tensor: ~0.4 MB")
    print("- After ILWT: ~1.6 MB") 
    print("- Hidden layers: ~1.6 MB each")
    print("- Model parameters: ~0.5-1 MB")
    print("\nTotal minimum: ~4-6 MB theoretical minimum")
    print("Actual usage is higher due to:")
    print("- Memory fragmentation")
    print("- CUDA runtime overhead")
    print("- Optimizer state")
    print("- Gradients storage")
    print("- Intermediate computation buffers")

if __name__ == "__main__":
    main()