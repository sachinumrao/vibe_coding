# api.py
import datetime
import os
import re

import soundfile as sf
import torch
from datasets import load_dataset  # For speaker embeddings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

app = FastAPI()

MUSIC_DIR = "music"
os.makedirs(MUSIC_DIR, exist_ok=True)

# --- Model Loading (Done once at startup) ---
MODEL_LOADING_ERROR = None
try:
    print("Loading SpeechT5 models... This might take a while on the first run.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    # Load xvector containing speaker's voice characteristics from a dataset
    print("Loading speaker embeddings...")
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    # Using a default speaker embedding (you can choose different speakers)
    # For example, 'slt' (female), 'bdl' (male) are common in CMU ARCTIC
    # Find an index for a speaker, e.g., by inspecting embeddings_dataset['speaker_id']
    # For simplicity, let's try to find 'slt' or use the first one.
    speaker_id_to_find = "slt"  # Example speaker
    speaker_embedding = None
    for i, sid in enumerate(embeddings_dataset["speaker_id"]):
        # The speaker_id in this dataset might be numeric.
        # We'll just pick one for demonstration. A real app might offer choices.
        # For now, let's just pick the first available one as a default.
        speaker_embedding = (
            torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0).to(device)
        )
        print(
            f"Using speaker embedding from index 0 (speaker_id: {embeddings_dataset[0]['speaker_id']})"
        )
        break

    if speaker_embedding is None:
        # Fallback if the specific speaker isn't found or logic is complex
        print("Default speaker not found, using the first available embedding.")
        speaker_embedding = (
            torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0).to(device)
        )

    print("Models and speaker embeddings loaded successfully.")

except Exception as e:
    MODEL_LOADING_ERROR = f"Failed to load SpeechT5 models: {e}"
    print(f"ERROR: {MODEL_LOADING_ERROR}")
    # To prevent app from starting if models fail to load critically:
    # raise RuntimeError(MODEL_LOADING_ERROR) # Or handle gracefully in endpoints

# --- End Model Loading ---


class TextToSpeechRequest(BaseModel):
    text: str


def sanitize_filename(text_snippet, max_length=50):
    s = re.sub(r"[^\w\s-]", "", text_snippet.lower())
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:max_length]


@app.post("/text-to-speech/")
async def convert_text_to_speech(request: TextToSpeechRequest):
    if MODEL_LOADING_ERROR:
        raise HTTPException(
            status_code=503, detail=f"Service Unavailable: {MODEL_LOADING_ERROR}"
        )
    if not hasattr(model, "generate_speech") or not hasattr(
        processor, "__call__"
    ):  # Basic check
        raise HTTPException(status_code=503, detail="TTS models not properly loaded.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        inputs = processor(text=request.text, return_tensors="pt").to(device)

        # Generate speech
        # Ensure speaker_embedding is not None and is on the correct device
        if speaker_embedding is None:
            raise HTTPException(status_code=500, detail="Speaker embedding not loaded.")

        with torch.no_grad():  # Important for inference
            speech = model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=speaker_embedding.to(device),
                vocoder=vocoder.to(device),
            )

        # The output is a tensor, convert to numpy array
        speech_numpy = speech.cpu().numpy()

        # Sanitize filename and create a filepath
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        text_snippet = sanitize_filename(
            request.text.split(".")[0] if "." in request.text else request.text
        )

        # Outputting as WAV as it's more direct from the model.
        # Streamlit can play WAV. If MP3 is a hard requirement, you'd need an extra step (e.g., pydub+ffmpeg).
        filename = f"{timestamp}_{text_snippet}.wav"  # Note: .wav extension
        filepath = os.path.join(MUSIC_DIR, filename)

        # Save as WAV file
        # The sample rate for SpeechT5 is typically 16000 Hz
        sample_rate = 16000  # Check model.config.sampling_rate if unsure, but SpeechT5 default is 16kHz
        sf.write(filepath, speech_numpy, samplerate=sample_rate)

        return {"message": f"Audio saved as {filename}", "filename": filename}
    except RuntimeError as e:  # Catch potential CUDA out of memory errors etc.
        if "out of memory" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail=f"Error generating audio: CUDA out of memory. Try shorter text or a smaller batch if applicable. Details: {str(e)}",
            )
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred during TTS: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # To run: uvicorn api:app --reload
    # Note: --reload might cause issues with GPU memory if models are reloaded frequently.
    # For production, run without --reload once stable.
    uvicorn.run(app, host="0.0.0.0", port=8000)
