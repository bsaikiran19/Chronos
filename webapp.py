import streamlit as st
import requests
import openai
import whisper
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
model = whisper.load_model("base")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def transcribe_audio(file_path):
    """Convert speech from an audio file into text."""
    result = model.transcribe(file_path)
    return result["text"]

def summarize_text(text):
    """Generate a concise summary from the transcript using GPT-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize this meeting transcript clearly and concisely."},
            {"role": "user", "content": text}
        ],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"].strip()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    """Handle file upload, perform transcription, and return the summary."""
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    transcript = transcribe_audio(file_path)
    summary = summarize_text(transcript)
    
    os.remove(file_path)
    return {"transcript": transcript, "summary": summary}

# Streamlit UI
st.set_page_config(page_title="Note Ninja - AI Meeting Assistant", page_icon="🎙️")
st.title("🎙️ Note Ninja - Your AI Meeting Assistant")
st.write("### Upload an audio file and let AI do the rest!")

API_URL = "http://127.0.0.1:8000/transcribe/"

uploaded_file = st.file_uploader("📂 Upload your audio file", type=["mp3", "wav", "m4a"]) 

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    st.write("💡 Click the button below to transcribe and summarize your meeting.")
    
    if st.button("🎤 Transcribe & Summarize"):
        with st.spinner("Processing your audio file... Please wait."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Done! Here’s what we got:")
                
                st.subheader("📜 Full Transcript:")
                st.text_area("", result.get("transcript", "No transcript available"), height=250)
                
                st.subheader("📌 AI-Generated Summary:")
                st.text_area("", result.get("summary", "No summary available"), height=180)
                
                st.write("✍️ Feel free to refine the summary or share it with your team!")
            else:
                st.error("⚠️ Something went wrong. Ensure the FastAPI server is running and try again.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
