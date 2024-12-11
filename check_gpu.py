import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")

    # Get number of available GPUs
    print("Number of GPUs:", torch.cuda.device_count())

    # Get the name of the current GPU
    print("Current GPU:", torch.cuda.get_device_name(0))

    # Get the current GPU memory usage
    free_mem, total_mem = torch.cuda.mem_get_info()
    print("Free GPU memory:", free_mem / 1024**2, "MB")
    print("Total GPU memory:", total_mem / 1024**2, "MB")

else:
    print("CUDA is not available")

# check if mps is available on Mac
if torch.backends.mps.is_available():
    print("MPS is available")
    num_devices = torch.mps.device_count()
    print("Number of MPS devices:", num_devices)
    allocated_mem = torch.mps.current_allocated_memory()
    print("Current MPS memory usage:", allocated_mem / 1024**2, "MB")
    driver_mem = torch.mps.driver_allocated_memory()
    print("Current MPS driver memory usage:", driver_mem / 1024**2, "MB")
    recommended_mem = torch.mps.recommended_max_memory()
    print("Recommended MPS memory usage:", recommended_mem / 1024**2, "MB")