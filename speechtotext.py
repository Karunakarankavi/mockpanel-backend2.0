import pyaudio
import websocket
import json
import threading
import time
import wave
from urllib.parse import urlencode
from datetime import datetime
from llmconnection import process_message
from flask import Flask, jsonify, request

from questionagent import get_question_endpoint
from texttospeech import ttsblend
from dotenv import load_dotenv
import os



# Load environment variables
load_dotenv()
api_key = os.getenv("ASSEMBLYAI_API_KEY")

CONNECTION_PARAMS = {
    "sample_rate": 16000,
    "format_turns": True,  # Request formatted final transcripts
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"

# Audio Configuration
FRAMES_PER_BUFFER = 800  # 50ms of audio (0.05s * 16000Hz)
SAMPLE_RATE = CONNECTION_PARAMS["sample_rate"]
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Global variables for audio stream and websocket
audio = None
stream = None
ws_app = None
audio_thread = None
ws_global = None
user_prompt =""
transcript = ""
stop_event = threading.Event()  # To signal the audio thread to stop

# WAV recording variables
recorded_frames = []  # Store audio frames for WAV file
recording_lock = threading.Lock()  # Thread-safe access to recorded_frames


# --- WebSocket Send Function ---

def send_to_assemblyai(data, is_binary=False):
    """
    Send data to AssemblyAI via WebSocket, unless stopmsgtollm is True.
    """
    global ws_global, stopmsgtollm



    if ws_global is None:
        print("WebSocket connection not established yet.")
        return False
    try:
        if is_binary:
            ws_global.send(data, websocket.ABNF.OPCODE_BINARY)
        else:
            if isinstance(data, dict):
                data = json.dumps(data)
            ws_global.send(data)
        return True
    except Exception as e:
        print(f"Error sending data to AssemblyAI: {e}")
        return False




def send_msg_to_llm(userid):
    """
    Flask API to send collected user_prompt to LLM
    """
    print("llm agent starting process ")
    global user_prompt, stopmsgtollm, transcript
    print(transcript , "transcript")

    # if not user_prompt.strip():
    #     return jsonify({"error": "No user prompt available"}), 400

    # Process with your LLM connection
    response = get_question_endpoint(transcript,userid)
    print(response.get("question"))
    user_prompt = ""
    blendtextdata = ttsblend(response.get("question"))
    stopmsgtollm = True
    return blendtextdata




# --- WebSocket Event Handlers ---
def on_open(ws):
    """Called when the WebSocket connection is established."""
    print("WebSocket connection opened.")
    global ws_global
    ws_global = ws  # Store the ws for use elsewhere

    # Start sending audio data in a separate thread
    def stream_audio():
        global stream
        print("Starting audio streaming...")
        while not stop_event.is_set():
            try:
                audio_data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)

                # Store audio data for WAV recording
                with recording_lock:
                    recorded_frames.append(audio_data)

                # Send audio data using the separate function
                send_to_assemblyai(audio_data, is_binary=True)
            except Exception as e:
                print(f"Error streaming audio: {e}")
                # If stream read fails, likely means it's closed, stop the loop
                break
        print("Audio streaming stopped.")

    global audio_thread
    audio_thread = threading.Thread(target=stream_audio)
    audio_thread.daemon = (
        True  # Allow main thread to exit even if this thread is running
    )
    audio_thread.start()


def on_message(ws, message):
    global user_prompt, transcript
    try:
        data = json.loads(message)
        msg_type = data.get('type')
        if msg_type == "Begin":
            session_id = data.get('id')
            expires_at = data.get('expires_at')
        elif msg_type == "Turn":
            transcript = data.get('transcript', '')
            formatted = data.get('turn_is_formatted', False)
            # Clear previous line for formatted messages
            if formatted:

                user_prompt+=transcript
            else:
                user_prompt += transcript
        elif msg_type == "Termination":
            audio_duration = data.get('audio_duration_seconds', 0)
            session_duration = data.get('session_duration_seconds', 0)
    except json.JSONDecodeError as e:
        print(f"Error decoding message: {e}")
    except Exception as e:
        print(f"Error handling message: {e}")


def on_error(ws, error):
    """Called when a WebSocket error occurs."""
    print(f"\nWebSocket Error: {error}")
    # Attempt to signal stop on error
    stop_event.set()


def on_close(ws, close_status_code, close_msg):
    """Called when the WebSocket connection is closed."""
    print(f"\nWebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")

    # Save recorded audio to WAV file
    save_wav_file()

    # Ensure audio resources are released
    global stream, audio
    stop_event.set()  # Signal audio thread just in case it's still running

    if stream:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        stream = None
    if audio:
        audio.terminate()
        audio = None
    # Try to join the audio thread to ensure clean exit
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=1.0)




# --- Main Execution ---
def run():
    global audio, stream, ws_app
    # Create WebSocketApp
    ws_app = websocket.WebSocketApp(
        API_ENDPOINT,
        header={"Authorization": api_key},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # Run WebSocketApp in a separate thread to allow main thread to catch KeyboardInterrupt
    ws_thread = threading.Thread(target=ws_app.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    try:
        # Keep main thread alive until interrupted
        while ws_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping...")
        stop_event.set()  # Signal audio thread to stop

        # Send termination message to the server using the new function
        terminate_message = {"type": "Terminate"}
        print(f"Sending termination message: {json.dumps(terminate_message)}")
        send_to_assemblyai(terminate_message)

        # Give a moment for messages to process before forceful close
        time.sleep(5)

        # Close the WebSocket connection (will trigger on_close)
        if ws_app:
            ws_app.close()

        # Wait for WebSocket thread to finish
        ws_thread.join(timeout=2.0)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        stop_event.set()
        if ws_app:
            ws_app.close()
        ws_thread.join(timeout=2.0)

    finally:
        # Final cleanup (already handled in on_close, but good as a fallback)
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
        if audio:
            audio.terminate()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    run()