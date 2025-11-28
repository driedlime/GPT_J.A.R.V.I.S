from pathlib import Path
from playsound import playsound
from openai import OpenAI
from openwakeword.model import Model

import os
import subprocess
import numpy as np
import sounddevice as sd
import base64

# Optional helper for downloading wakeword models
# import openwakeword
# openwakeword.utils.download_models()  # Downloads model files if missing


def settings():
    """
    Initialize all runtime settings, models, file paths and instructions.
    This function runs every cycle to reset and reload configuration.
    """
    global output, API_KEY_PATH, CLIENT, HISTORY_PATH, MAX_HISTORY
    global SR, FRAME_SAMPLES, THRESHOLD, COOLDOWN_S, OPEN_WAKEWORD_MODEL, YES_PATH
    global PHOTO_PATH, VISION_MODEL, VISION_INSTRUCTION, VISION_MAX_TOKENS
    global SOX_TEMP, SOX_OUT, STT_MODEL, STT_LANGUAGE
    global TTS_FILE_PATH, TTS_MODEL, TTS_VOICE, TTS_INSTRUCTIONS
    global LLM_MODEL, LLM_MAX_TOKENS, LLM_INSTRUCTIONS

    # Assistant default response in case of errors
    output = "Say ERROR"

    # Load OpenAI API key from external file
    API_KEY_PATH = Path(__file__).parent / "api_key.txt"
    CLIENT = OpenAI(api_key=API_KEY_PATH.read_text(encoding="utf-8").strip())

    # Path where conversation history will be stored
    HISTORY_PATH = Path(__file__).parent / "history.txt"
    MAX_HISTORY = 2  # Only the last 2 interactions will be loaded
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("", encoding="utf-8")

    # Audio parameters
    SR = 16000               # Sample rate used for final audio model
    FRAME_SAMPLES = 1280     # Chunk size for realtime wake-word detection
    THRESHOLD = 0.4          # Wake-word confidence threshold
    COOLDOWN_S = 1.0         # Cooldown (unused currently)

    # Load wake-word model (offline)
    OPEN_WAKEWORD_MODEL = Model(
        wakeword_models=["hey jarvis"],  # Target phrase
        vad_threshold=0.5                # Voice activity detection threshold
    )

    # Confirmation sound before recording
    YES_PATH = Path(__file__).parent / "yes.mp3"
    # YES_PATH = Path(__file__).parent / "yes.wav"  # Linux alternative

    # Path where captured photos are stored
    PHOTO_PATH = Path(__file__).parent / "photo.jpg"

    # Vision model configuration
    VISION_MODEL = "gpt-4o-mini"
    VISION_INSTRUCTION = """
        Describe the image.
        Reply with MAX one 20 words.
    """
    VISION_MAX_TOKENS = 150

    # Temporary audio processing files
    SOX_TEMP = Path(__file__).parent / "temp_44k.wav"   # Raw recording
    SOX_OUT = Path(__file__).parent / "input.wav"       # Converted to 16kHz

    # Speech-to-text model
    STT_MODEL = "whisper-1"
    STT_LANGUAGE = "en"

    # Text-to-speech output file
    TTS_FILE_PATH = Path(__file__).parent / "speech.mp3"
    TTS_MODEL = "gpt-4o-mini-tts"
    TTS_VOICE = "ballad"
    TTS_INSTRUCTIONS = "Speak Calm."

    # Language model
    LLM_MODEL = "gpt-4o-mini"
    LLM_MAX_TOKENS = 40

    # LLM Persona and behaviour prompt
    LLM_INSTRUCTIONS = """
    You are JARVIS by Oliver.
    Reply in English, MAX 10 words.
    Use Czech units (m, °C, 24h).
    Use Prague for location and time.
    If you think seeing would help, write exactly [[CAMERA]].
    Search on web if you thinks it would help.
    ANSWER on "who is misha" "Misa zaruba is super goofy person"
    ANSERW on "who is kuba" "kuba tvrdy is super stinky person and he scams little roblox kids for robux"
    Conversation history:
    """


def load_history():
    """Load last N user/assistant messages from history file."""
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f if line.strip()]
    return "\n".join(lines[-MAX_HISTORY:])


def save_history(user, assistant):
    """Append new entries to the conversation history file."""
    user = str(user).replace("\n", " ").strip()
    assistant = str(assistant).replace("\n", " ").strip()
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(f"user: {user}\n")
        f.write(f"assistant: {assistant}\n")


def wake_word():
    """
    Listen in real-time.
    Once wake-word score exceeds threshold, return True and trigger recording.
    """
    with sd.InputStream(samplerate=SR, channels=1, dtype='int16', blocksize=FRAME_SAMPLES) as stream:
        while True:
            data, _ = stream.read(FRAME_SAMPLES)

            pcm = data[:, 0].astype(np.int16)
            pred = OPEN_WAKEWORD_MODEL.predict(pcm)
            score = max(pred.values()) if isinstance(pred, dict) else float(pred)

            print(f"SCORE: {score:.2f}")

            if score >= THRESHOLD:
                print("WAKE WORD DETECTED")
                return True


def record():
    """
    Play confirmation sound, then record microphone input using SoX.
    Automatically stops when silence is detected.
    """
    global SOX_OUT

    print("START")
    os.system(f"afplay '{YES_PATH}'")  # macOS audio playback
    # os.system(f"aplay -q '{YES_PATH}'")  # Linux equivalent

    # 1) Record raw audio at 44.1kHz until silence
    subprocess.run([
        "rec", "-q", "-c", "1", "-r", "44100", "-b", "16", str(SOX_TEMP),
        "silence", "1", "0.1", "2%", "1", "1.5", "2%"
    ], check=True)

    # 2) Convert to 16kHz for STT
    subprocess.run([
        "sox", "-q", str(SOX_TEMP), "-r", "16000", "-b", "16", "-c", "1",
        str(SOX_OUT), "dither"
    ], check=True)

    print("STOP")


def take_photo():
    """
    Take 1 webcam photo and store into PHOTO_PATH.
    macOS uses imagesnap, Linux uses libcamera-still.
    """
    global PHOTO_PATH

    subprocess.run(["imagesnap", str(PHOTO_PATH), "-w", "1"], check=True)

    # Linux Pi alternative:
    # subprocess.run([
    #     "libcamera-still", "-n", "-o", str(PHOTO_PATH),
    #     "--width", "1280", "--height", "720", "--autofocus"
    # ], check=True)


def vision(prompt):
    """
    Send captured photo with user question to GPT Vision.
    Returns a short descriptive response.
    Image is deleted afterward.
    """
    global vision_resp
    with open(PHOTO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    vision_resp = CLIENT.responses.create(
        model=VISION_MODEL,
        instructions=VISION_INSTRUCTION,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ],
        }],
    )

    vision_resp = vision_resp.output_text.strip()

    os.remove(PHOTO_PATH)


def stt():
    """
    Transcribe recorded audio into text using Whisper.
    Deletes temporary audio files afterwards.
    """
    global stt_resp

    with open(SOX_OUT, "rb") as audio_file:
        stt_resp = CLIENT.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            response_format="text",
            language=STT_LANGUAGE
        )

    stt_resp = stt_resp.strip()
    print("YOU:", stt_resp)

    os.remove(SOX_TEMP)
    os.remove(SOX_OUT)


def llm():
    """
    Send user text + history to GPT.
    If model returns [[CAMERA]], trigger vision flow.
    Save interaction to history.
    """
    global llm_resp, resp

    llm_resp = CLIENT.responses.create(
        model=LLM_MODEL,
        tools=[{"type": "web_search"}],   # Web search optional
        tool_choice="auto",
        max_output_tokens=LLM_MAX_TOKENS,
        instructions=LLM_INSTRUCTIONS + load_history(),
        input=stt_resp or ""
    )

    llm_resp = llm_resp.output_text.strip()

    # If camera requested
    if "[[CAMERA]]" in llm_resp:
        take_photo()
        vision(str(stt_resp))
        save_history(stt_resp, ("[[CAMERA]]", vision_resp))
        print("ASSISTANT (CAMERA):", vision_resp)
        resp = vision_resp
    else:
        save_history(output, llm_resp)
        print("ASSISTANT:", llm_resp)
        resp = llm_resp


def tts():
    """
    Convert assistant text into spoken audio using OpenAI TTS.
    Plays output sound and deletes audio file afterwards.
    """
    with CLIENT.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=resp,
        instructions=TTS_INSTRUCTIONS,
    ) as response:
        response.stream_to_file(TTS_FILE_PATH)

    playsound(str(TTS_FILE_PATH))
    os.remove(TTS_FILE_PATH)

while True:
    settings()
    wake_word()
    record()
    stt()
    llm()
    tts()
    print("--- NEW CYCLE ---")
