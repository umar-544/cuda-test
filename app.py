import torch
import time

# check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. Please enable it or run on a system with a supported GPU.")
    exit(1)
# matrix size
size = 1000


# generate random matrices
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)


# CPU timing
start_cpu = time.time()
result_cpu = torch.mm(a_cpu, b_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print(f"CPU time: {cpu_time:.6f} seconds")


# move tensors to GPU
a_gpu = a_cpu.to("cuda")
b_gpu = b_cpu.to("cuda")


# warm up gpu (as first call can be slower)
torch.mm(a_gpu, b_gpu)


# GPU timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
result_gpu = torch.mm(a_gpu, b_gpu)
end_event.record()


# wait for GPU to finish
torch.cuda.synchronize()
gpu_time = start_event.elapsed_time(end_event) / 1000
print(f"GPU time: {gpu_time:.6f} seconds")

# difference in speed
print(f"difference in speed: {cpu_time / gpu_time:.2f}x")