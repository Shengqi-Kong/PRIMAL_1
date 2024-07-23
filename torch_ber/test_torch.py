import torch
cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

# Print CUDA devices
print("CUDA Devices:")
for i, device in enumerate(cuda_devices):
    print(f"Device {i}: {device}")

# Check if CUDA is available
if torch.cuda.is_available():
    print("\nCUDA is available")
else:
    print("\nCUDA is not available")