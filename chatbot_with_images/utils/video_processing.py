import os
import requests
import tiktoken
import time
import cv2, faiss
import torch
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import List
from pprint import pprint
from itertools import cycle
from multiprocessing import Pool
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL



def worker_get_answer(args):
    idx, prompt, api_key = args
    output_text, status_code = get_answer(prompt, api_key)
    if status_code == 200:
        return (idx, output_text)
    else:
        return (idx, f"Error: {status_code}")

def batch_get_answer(prompts, key_gemini, max_concurrent=5):
    args_list = [(i, prompt, next(key_gemini)) for i, prompt in enumerate(prompts)]

    with Pool(processes=max_concurrent) as pool:
        results = pool.map(worker_get_answer, args_list)

    return results


def get_video_id(url):
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    else:
        raise ValueError("URL không hợp lệ hoặc thiếu video ID.")

def extract_transcript(url):
    video_id = get_video_id(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi', 'en'])
    transcript_dict = {}
    for entry in transcript:
        timestamp = entry['start']
        hours, remainder = divmod(int(timestamp), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        text = entry['text']
        transcript_dict[time_str] = text
    return transcript_dict



def count_tokens(extracted):
    # Token của GPT gần tương tự với Gemini với nhiều hơn khoảng 7%
    tokenizer = tiktoken.get_encoding("o200k_base")

    total_tokens = 0
    for _, text in extracted.items():
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
    return total_tokens


# pprint(extracted)
def get_video_info(url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)       
        title = info_dict.get('title', 'N/A')
        views = info_dict.get('view_count', 'N/A')
        llm_input = (
            f"Title: {title}\n"
            f"Views: {views}\n"
        )

    return llm_input

def get_formatted_transcript(transcript, info):
    format_transcript = ""
    for timestamp, text in transcript.items():
        format_transcript += f"{timestamp}: {text}\n"
    formatted = info + "\n" + format_transcript
    return formatted

def init_processor():
    global image_processor, model, device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Device using now:", device)
    image_processor = AutoImageProcessor.from_pretrained('models/dinov2-base')
    model = AutoModel.from_pretrained('models/dinov2-base').to(device)

def download_video_ytdlp(url, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Define the options for yt-dlp
    ydl_opts = {
        'format': 'mp4/best',  # Specify the format to download
        'outtmpl': os.path.join(output_dir, 'downloaded_video.mp4'),  # Output template
        'quiet': True  # Suppress output
    }
    
    # Download the video
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.join(output_dir, 'downloaded_video.mp4')


def extract_frames_and_build_index(video_url, interval_seconds=2):
    import cv2
    output_directory = os.path.join(os.getcwd(), "videos")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        pass

    video_path = download_video_ytdlp(video_url, output_directory)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    all_image_embeddings = []
    all_image_paths = []

    print("\n Extracting frames and building index...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= total_frames:
            break  # Kết thúc vòng lặp khi đọc hết frame hoặc vượt quá tổng số frame

        if frame_count % frame_interval == 0:
            # Xử lý frame và lưu lại thành file ảnh
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            total_seconds = int(frame_count / fps)
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            timestamp = f"{str(hours).zfill(2)}_{str(minutes).zfill(2)}_{str(seconds).zfill(2)}"
            frame_path = os.path.join(os.getcwd(), f"videos/frame_{timestamp}.jpg")
            # img.save(frame_path)
            # Trích xuất đặc trưng từ ảnh
            with torch.no_grad():
                inputs = image_processor(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
            
            # Tính vector
            features = outputs.last_hidden_state.mean(dim=1)
            vector = features.detach().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(vector)

            all_image_embeddings.append(vector)
            all_image_paths.append(frame_path)

            print(f"✅ Processed frame at {timestamp} seconds")

        frame_count += 1

    cap.release()

    if all_image_embeddings:
        all_image_embeddings = np.vstack(all_image_embeddings)
        index = faiss.IndexFlatL2(all_image_embeddings.shape[1])
        index.add(all_image_embeddings)
        index.image_paths = all_image_paths

        faiss.write_index(index, "vector.index")

        with open("image_paths.json", "w") as f:
            json.dump(all_image_paths, f)

        print(f"\n Index built and saved successfully with {len(all_image_paths)} frames.")



countinue_prompt = """
You are an assistant tasked with answering questions based on the provided context, which is a transcript of a YouTube video.
Use the given context to answer the user's query in a clear, concise, and accurate manner.

Context (YouTube Video Transcript):
{transcript}
User Query:
{user_query}
Answer:
"""

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=9RhWXPcKBI8"
    # youtube_url = "https://www.youtube.com/watch?v=7xTGNNLPyMI&t=1937s"
    st = time.perf_counter()
    extracted = extract_transcript(youtube_url)
    questions = ["What is the main topic of the video?", "Summary of the video", "what is the prize?", "get the sentence contain the word 'tesla' and time stamp"]
    info = get_video_info(youtube_url)
    formatted_transcript = get_formatted_transcript(extracted, info=info)
    et = time.perf_counter()
    print(f"Time taken: {et - st} seconds")