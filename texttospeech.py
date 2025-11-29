import io
import base64
from flask import Flask, request, jsonify
from pydub import AudioSegment
from google.cloud import texttospeech
from getphenome import generate_phonemes


app = Flask(__name__)
client = texttospeech.TextToSpeechClient.from_service_account_file("gcpkey.json")

def ttsblend(text):
    if not text:
        return jsonify({"error": "Text is required"}), 400

    # 1️⃣ Generate audio from Google TTS
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-D"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    # 2️⃣ Convert audio bytes to AudioSegment to get duration
    audio_bytes = io.BytesIO(response.audio_content)
    audio = AudioSegment.from_file(audio_bytes, format="mp3")
    duration_seconds = audio.duration_seconds

    # 3️⃣ Generate blendData using your LLM function
    blendData = generate_phonemes(text , duration_seconds)

    # 4️⃣ Encode audio to base64 for JSON transport
    audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
    print(duration_seconds)
    # 5️⃣ Return combined JSON
    return jsonify({
        "audioSource": audio_base64,  # frontend can decode base64 to play
        "blendData": blendData,
        "duration" : duration_seconds,
        "question" : text
    })

if __name__ == "__main__":
    app.run(port=3001, debug=True)
