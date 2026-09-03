#!/usr/bin/env python3
"""
DDP Diagnostic script to help identify and solve issues
"""

import torch
import psutil
import GPUtil
import os

def check_system_resources():
    """Check system resources that might affect DDP"""
    print("🔍 System Resource Check")
    print("=" * 50)

    # CPU info
    print(f"CPU cores: {psutil.cpu_count()}")
    print(".1f")
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")

    # GPU info
    if torch.cuda.is_available():
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("❌ CUDA not available!")

    # Check for existing processes
    print("Process Check")
    print("-" * 30)

    # Check for existing Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    print(f"Found {len(python_processes)} Python processes")
    for proc in python_processes[:5]:  # Show first 5
        print(f"  PID {proc['pid']}: {proc['name']}")

def check_network_connectivity():
    """Check network connectivity for DDP"""
    print("Network Check")
    print("-" * 30)

    import socket
    try:
        # Test localhost connectivity
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 12345))
        sock.close()

        if result == 0:
            print("⚠️  Port 12345 is in use!")
        else:
            print("✅ Port 12345 is available")

    except Exception as e:
        print(f"❌ Network check failed: {e}")

def recommend_settings():
    """Provide recommendations for DDP settings"""
    print("Recommendations")    
    print("-" * 30)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)) / 1024**3

        print(f"GPU Count: {gpu_count}")
        print(f"Total memory: {total_memory:.1f} GB")
        # Recommend batch size based on GPU memory
        if total_memory > 40:  # High-end GPUs
            recommended_batch = 8
        elif total_memory > 20:  # Mid-range GPUs
            recommended_batch = 4
        else:  # Low-end GPUs
            recommended_batch = 2

        print(f"Recommended batch_size per GPU: {recommended_batch}")
        print(f"Total effective batch size: {recommended_batch * gpu_count}")

        # Memory per sample estimate (rough)
        print("Memory considerations:")
        print("  - Monitor GPU memory usage during training")
        print("  - Reduce batch_size if you see OOM errors")
        print("  - Consider gradient_checkpointing for large models")

def main():
    """Main diagnostic function"""
    print("🚀 DDP Diagnostic Tool")
    print("=" * 50)

    check_system_resources()
    check_network_connectivity()
    recommend_settings()

    print("Next Steps:")    
    print("1. If you see port conflicts, kill existing processes or change ports")
    print("2. If memory is low, reduce batch_size in your config")
    print("3. Run with --log-level=INFO for more detailed logging")
    print("4. Use `watch nvidia-smi` to monitor GPU usage in real-time")

if __name__ == "__main__":
    main()
