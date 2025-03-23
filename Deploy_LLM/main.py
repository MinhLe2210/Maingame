from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import time
from openai import OpenAI

app = FastAPI()

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

class PromptRequest(BaseModel):
    prompts: List[str]
    model: Optional[str] = "meta-llama/Llama-3.1–8B-Instruct"
    max_tokens: int = 100

results = {}

async def generate_completion(prompt, model, max_tokens):
    start_time = time.time()
    
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stream=False
    )
    
    first_token_time = time.time() - start_time
    completion_time = time.time() - start_time
    tokens_generated = completion.usage.completion_tokens
    
    throughput = tokens_generated / completion_time if completion_time > 0 else 0
    
    result = {
        "prompt": prompt,
        "completion": completion.choices[0].text.strip(),
        "time_to_first_token": first_token_time,
        "total_time": completion_time,
        "throughput": throughput,
        "tokens_generated": tokens_generated
    }
    
    return result

async def handle_batch(prompts, model, max_tokens):
    tasks = [
        generate_completion(prompt, model, max_tokens)
        for prompt in prompts
    ]
    
    results = await asyncio.gather(*tasks)
    return results

@app.post("/generate")
async def generate(request: PromptRequest, background_tasks: BackgroundTasks):
    if not request.prompts:
        raise HTTPException(status_code=400, detail="Prompts cannot be empty.")
    
    background_tasks.add_task(
        lambda: asyncio.run(handle_batch(request.prompts, request.model, request.max_tokens))
    )
    
    output = await handle_batch(request.prompts, request.model, request.max_tokens)
    
    request_id = str(time.time())
    results[request_id] = output
    
    return {
        "request_id": request_id,
        "status": "completed",
        "results": output
    }

@app.get("/results/{request_id}")
async def get_results(request_id: str):
    if request_id not in results:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    
    return results[request_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)
