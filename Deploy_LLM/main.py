import requests
import json
import time
import torch

# API URL
url = "http://localhost:30080/completions"

# Request payload
payload = {
    "model": "meta-llama/Llama-3.1–8B-Instruct",
    "prompt": "Once upon a time,",
    "max_tokens": 100
}

# Measure memory usage before inference
if torch.cuda.is_available():
    torch.cuda.synchronize()
    vram_before = torch.cuda.memory_allocated()
else:
    vram_before = None

# Send request and measure time to first token and throughput
start_time = time.time()
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), stream=True)

# Time to first token
first_token_time = None
total_tokens = 0
start_first_token = time.time()

for line in response.iter_lines():
    if line:
        total_tokens += 1
        if first_token_time is None:
            first_token_time = time.time() - start_first_token

# Total completion time
end_time = time.time()
total_time = end_time - start_time

# Throughput (tokens per second)
throughput = total_tokens / total_time if total_time > 0 else 0

# Measure VRAM usage after inference
if torch.cuda.is_available():
    torch.cuda.synchronize()
    vram_after = torch.cuda.memory_allocated()
    vram_used = vram_after - vram_before
else:
    vram_used = None

# Report metrics
metrics = {
    "Time to First Token (s)": first_token_time,
    "Total Tokens Generated": total_tokens,
    "Total Time (s)": total_time,
    "Throughput (tokens/sec)": throughput,
    "VRAM Used (bytes)": vram_used if vram_used is not None else "N/A"
}

# Print the report
print("\n=== Model Performance Report ===")
for key, value in metrics.items():
    print(f"{key}: {value}")
