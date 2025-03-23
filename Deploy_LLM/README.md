

## 1. **Deploy Model with vLLM**
You can deploy the model using **vLLM** to serve an API endpoint that can call an LLM to generate answers:

```bash
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model bigscience/bloomz-1b1
```

- This command will launch an API server using the **bigscience/bloomz-1b1** model.
- The server will be available at `http://localhost:8000/v1/chat/completions`.

---

## 2. **Batch Processing Queries**
You can use `main.py` to process multiple queries in batches:

```bash
python main.py
```

---

## 3. **Example: Batch Processing with FastAPI**
Here's an example of how to send batch requests using `curl`:

```bash
curl -X POST "http://localhost:8123/generate" \
-H "Content-Type: application/json" \
-d '{
  "prompts": [
    "Describe a futuristic city in the year 2150.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "Explain the concept of recursion with an example in Python."
  ],
  "model": "bigscience/bloomz-1b1",
  "max_tokens": 100
}'
```

- This example sends three prompts to the server.
- The server will generate responses using the specified model and return the results in a single batch.

---

## 4. **Evaluation**
### 4.1. **Hardware Setup**
- **GPU:** NVIDIA RTX 3090 (via Vast.ai)  
- **VRAM Usage:** 17.5 GiB / 24 GiB  

---

### 4.2. **Model Performance Report**
| Metric | Value |
|--------|-------|
| **Total Prompts** | 21 |
| **Total Time (s)** | 0.91 |
| **Average Time per Prompt (s)** | 0.0432 |
| **Average Time to First Token (s)** | 0.0432 |
| **Average Throughput (tokens/s)** | 133.01 |
| **Total Tokens Generated** | 146 |

---

### 4.3. **Summary**
- The model processed **21 prompts** in approximately **0.91 seconds**.  
- The average time to process a single prompt was **0.0432 seconds**.  
- The average time to first token was **0.0432 seconds**, indicating minimal latency.  
- The model achieved an impressive throughput of **133 tokens per second**, demonstrating high processing efficiency on GPU.  



## 5. **Running on CPU or Using Quantized Models**
If you want to run the model on CPU or use a quantized version for better efficiency, you can use **Llama.cpp**:

### 5.1. **Example: Running a Quantized Model with Llama.cpp**
```bash
./llama.cpp -m models/your-model-name.bin -n 100
```

- `models/your-model-name.bin` – Path to the quantized model file.  
- `-n 100` – Number of tokens to generate.  
- `llama.cpp` is optimized for CPU-based inference and can significantly reduce memory usage and improve latency.  

---

**Future Optimizations:**
- Consider using GPU-based inference for faster processing.  
- Fine-tune the model for specific domains to improve response quality.  
- Explore mixed-precision inference to balance speed and accuracy.  



