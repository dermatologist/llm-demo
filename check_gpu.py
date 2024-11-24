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