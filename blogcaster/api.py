# api.py
import datetime
import os
import re

from fastapi import FastAPI, HTTPException
from gtts import gTTS
from pydantic import BaseModel

app = FastAPI()

MUSIC_DIR = "./music"
os.makedirs(MUSIC_DIR, exist_ok=True)


class TextToSpeechRequest(BaseModel):
    text: str


def sanitize_filename(text_snippet, max_length=50):
    # Remove special characters, replace spaces with underscores
    s = re.sub(r"[^\w\s-]", "", text_snippet.lower())
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:max_length]


@app.post("/text-to-speech/")
async def convert_text_to_speech(request: TextToSpeechRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        # Generate a unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use a snippet of the text for a more descriptive filename
        text_snippet = sanitize_filename(
            request.text.split(".")[0] if "." in request.text else request.text
        )
        filename = f"{timestamp}_{text_snippet}.mp3"
        filepath = os.path.join(MUSIC_DIR, filename)

        tts = gTTS(text=request.text, lang="en", slow=False)
        tts.save(filepath)

        return {"message": f"Audio saved as {filename}", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # To run: uvicorn api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
