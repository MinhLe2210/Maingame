
```markdown
# Chatbot with Videos Pipeline

## How to Use
```bash
pip install uv
uv venv .venv --python=python3.12
.venv/Scripts/activate

uv pip install -r requirements.txt
```

Once installed, you can run the code in `utils/chatbot.ipynb`.


## 1. Workflow of Chatbot
### 1.1. **User Input Handling**
- When a user enters a query, the system will first analyze it using an LLM (Large Language Model) to determine whether the query contains a YouTube URL or not.
    - **If no YouTube URL is present:**  
      → The system will proceed to answer the question using only the LLM.

    - **If a YouTube URL is present:**  
      → The system will check if the user also wants to use image-based input or not.


### 1.2. **Video Processing**
#### 1.2.1. **If the input contains a YouTube URL and no image input:**
- The system will process the video URL as follows:
    - **Groq API + Whisper Large v3:**  
      - I initially used Groq API with Whisper Large Turbo v3 to transcribe audio to text. It works well with small files but fails with large files (e.g., a 27-minute video returns a "file too large" error).
      
    - **Youtube Transcript API:**  
      - I also tested the `youtube-transcript-api` library, but it has a drawback of being limited to only **2 requests per minute**.

    - **Solution:**  
      - To overcome these limitations, I'm considering hosting my own audio-to-text service using Whisper. This would allow me to generate transcripts for long YouTube videos without external API limitations.

    - **Once transcript generation is complete:**  
      - I feed the transcript into **Gemini Flash 2.0**, which supports up to **2 million input tokens** — sufficient to handle long transcripts.  
      *(My current infrastructure cannot run LLM locally for testing due to hardware limitations. To optimize costs, I might consider running the model locally and fine-tuning it in the future for better performance on proprietary data.)*


#### 1.2.2. **If the input contains a YouTube URL and image input:**
- The system will process audio as described above.
- Additionally, it will extract **one frame every 2 seconds** from the video and embed them using **DINO-v2**.
    - I store the embeddings in an in-memory store like **Redis** or **Faiss**.  
    - I'm currently using **Faiss** because it provides better performance in production.  
        - **CPU indexing:** ~0.7 seconds per image.  
        - **GPU indexing:** Over **100 images per second**.

- For retrieval:
    - The system computes cosine similarity between the user's input image and stored embeddings to find the most relevant frame.
    - The timestamp of the retrieved image is also saved with the embedding for reference.
    - For better accuracy, I could integrate a **reranker** like **ColGwen** (a multimodal LLM) to evaluate whether the retrieved image is truly relevant to the input query.


### 1.3. **Generating the Final Response**
- After retrieving the transcript and the most relevant frames:
    - The system feeds both the transcript and the images (with timestamps) into the LLM to generate the final response.  
    - If **Gemini** is not suitable, I could consider hosting a multimodal LLM like:
        - **Vistral** (from Mistral AI)  
        - **GwenVL** (from Alibaba)  

*(In the code, the indexing process is already implemented. For testing, I processed a 27-minute video in ~360 seconds. In production, this can be accelerated further by storing embeddings in RAM instead of a directory.)*


✅ **Future Optimizations:**
- Hosting a local Whisper instance to avoid file size limitations.
- Fine-tuning the LLM to improve performance with custom data.
- Adding a reranker like **ColGwen** to improve the relevance of image-based search.

