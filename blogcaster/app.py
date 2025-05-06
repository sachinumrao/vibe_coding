# app.py
import datetime
import os
import re

import requests
import streamlit as st

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/text-to-speech/"
MUSIC_DIR = "music"
os.makedirs(MUSIC_DIR, exist_ok=True)


# --- Helper Functions ---
def get_audio_files():
    """Lists .wav (and optionally .mp3) files in the MUSIC_DIR, sorted by modification time."""
    if not os.path.exists(MUSIC_DIR):
        return []
    try:
        # Now primarily looking for .wav files, but can include .mp3 for backward compatibility
        files = [f for f in os.listdir(MUSIC_DIR) if f.endswith((".wav", ".mp3"))]
        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(MUSIC_DIR, f)), reverse=True
        )
        return files
    except Exception as e:
        st.sidebar.error(f"Error listing audio files: {e}")
        return []


# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Local Text & Audio App", layout="wide", initial_sidebar_state="expanded"
)

# --- Sidebar for Audio Playback ---
st.sidebar.title("🎧 Audio Player")
st.sidebar.caption("Listen to generated audio (Hugging Face Model)")  # Updated caption

audio_files_list = get_audio_files()

if not audio_files_list:
    st.sidebar.info("No audio files found yet. Generate some from the main panel!")
else:
    selected_audio_file = st.sidebar.selectbox(
        "⬇️ Select an audio file:", audio_files_list, key="audio_select_sidebar"
    )
    if selected_audio_file:
        audio_path = os.path.join(MUSIC_DIR, selected_audio_file)
        try:
            with open(audio_path, "rb") as ap:
                # Determine format based on extension
                file_extension = os.path.splitext(selected_audio_file)[1].lower()
                audio_format = (
                    "audio/wav" if file_extension == ".wav" else "audio/mpeg"
                )  # mpeg for mp3
                st.sidebar.audio(ap.read(), format=audio_format)
            st.sidebar.markdown(f"Playing: **{selected_audio_file}**")
        except FileNotFoundError:
            st.sidebar.error(f"Error: File '{selected_audio_file}' not found.")
        except Exception as e:
            st.sidebar.error(f"Could not play audio: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Using local Hugging Face TTS model.")


# --- Main Area for Text to Audio ---
st.title("📝 Blogcaster")  # Updated title
st.caption("Convert text to speech using a local Hugging Face model")

if "text_area_content" not in st.session_state:
    st.session_state.text_area_content = ""

text_input = st.text_area(
    "Paste your text here:",
    value=st.session_state.text_area_content,
    height=200,
    key="main_text_area",
)
st.session_state.text_area_content = text_input

st.markdown(
    "<p style='text-align: center; font-weight: bold;'>OR</p>", unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a .txt file", type=["txt"], key="main_file_uploader"
)

final_text_to_convert = ""
source_of_text = ""

if uploaded_file is not None:
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        if st.checkbox(
            "Use uploaded file content? (Replaces text area)", key="use_file_cb"
        ):
            final_text_to_convert = string_data
            source_of_text = f"uploaded file '{uploaded_file.name}'"
            if (
                st.session_state.text_area_content != string_data
            ):  # Only update and rerun if content changed
                st.session_state.text_area_content = string_data
                st.experimental_rerun()
        elif text_input.strip():
            final_text_to_convert = text_input
            source_of_text = "text area"
        else:
            final_text_to_convert = string_data
            source_of_text = (
                f"uploaded file '{uploaded_file.name}' (text area was empty)"
            )
    except Exception as e:
        st.error(f"Error reading file: {e}")
elif text_input.strip():
    final_text_to_convert = text_input
    source_of_text = "text area"

if st.button("🔊 Convert to Audio (Local Model)", key="main_submit_button"):
    if not final_text_to_convert.strip():
        st.warning(
            "⚠️ Please enter text or upload a .txt file and select 'Use uploaded file content'."
        )
    else:
        st.info(f"Converting text from: {source_of_text} using local model.")
        with st.spinner(
            "⚙️ Converting text to audio (local model)... This can take some time."
        ):
            try:
                payload = {"text": final_text_to_convert}
                # Local models can take longer, especially on CPU or for longer text
                response = requests.post(
                    API_URL, json=payload, timeout=300
                )  # Increased timeout
                response.raise_for_status()
                result = response.json()
                returned_filename = result.get("filename", "audio.wav")
                st.success(
                    f"✅ Success! Audio saved as '{returned_filename}' in the 'music' folder."
                )
                st.balloons()
                # st.session_state.text_area_content = "" # Optional: clear input
                st.experimental_rerun()  # Refresh audio list in sidebar
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Connection Error: Could not connect to the API. Is the FastAPI server running at {API_URL}?"
                )
            except requests.exceptions.HTTPError as e:
                error_detail = "Could not retrieve error details from API."
                try:
                    error_detail = e.response.json().get("detail", e.response.text)
                except:  # pylint: disable=bare-except
                    pass
                st.error(f"❌ API Error: {e.response.status_code} - {error_detail}")
            except requests.exceptions.Timeout:
                st.error(
                    "❌ API Error: The request timed out. The text might be too long, or the local model is slow (especially on CPU)."
                )
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
