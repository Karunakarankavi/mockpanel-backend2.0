import asyncio
import threading
import websockets
from flask import Flask, request, jsonify
from llmconnection import process_message
from speechtotext import send_to_assemblyai, run, send_msg_to_llm
from flask_cors import CORS
import subprocess
import time

# ------------------- WebSocket Handler -------------------
async def handler(websocket):
    print("🔗 Client connected")
    global stopmsgtollm
    try:
        async for message in websocket:
            if not stopmsgtollm:
                if isinstance(message, (bytes, bytearray)):
                    send_to_assemblyai(message, is_binary=True)
                else:
                    send_to_assemblyai(message)
            else:
                print("⚠️ Message to LLM is stopped (stopmsgtollm=True)")
    except websockets.exceptions.ConnectionClosed as e:
        print("❌ Client disconnected:", e)


# ------------------- Flask API -------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

stopmsgtollm = False

@app.route("/send-msg", methods=["POST"])
def send_msg_api():
    data = request.get_json()
    user_id = data.get("userId")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    response = send_msg_to_llm(user_id)
    return response

@app.route("/reconnect", methods=["POST"])
def reconnect():
    global stopmsgtollm
    stopmsgtollm = False
    return jsonify({"success": True, "stopmsgtollm": stopmsgtollm})


def run_flask():
    print("🚀 Flask API started at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)


# ------------------- Start Both -------------------
async def main():
    print("⚙️ Starting extractresume.py...")
    extract_process = subprocess.Popen(["python", "extractresume.py"])
    time.sleep(3)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    stt_thread = threading.Thread(target=run, daemon=True)
    stt_thread.start()

    async with websockets.serve(handler, "localhost", 8001):
        print("✅ WebSocket server started at ws://localhost:8001")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
