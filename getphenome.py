import numbers
import os
import json
import time
from flask import Flask, request, jsonify
from phonemizer import phonemize
import re


app = Flask(__name__)

# Your phenome map
phenome_map = {
    "p":   { "jawOpen": 0.2, "mouthFunnel": 0.8, "mouthPucker": 0.6, "tongue_out": 0.0, "tongue_up": 0.0 },
    "b":   { "jawOpen": 0.2, "mouthFunnel": 0.8, "mouthPucker": 0.6, "tongue_out": 0.0, "tongue_up": 0.0 },
    "m":   { "jawOpen": 0.2, "mouthFunnel": 0.7, "mouthPucker": 0.7, "tongue_out": 0.0, "tongue_up": 0.0 },
    "t":   { "jawOpen": 0.3, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.8 },
    "d":   { "jawOpen": 0.3, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.8 },
    "n":   { "jawOpen": 0.3, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.7 },
    "k":   { "jawOpen": 0.4, "mouthFunnel": 0.0, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.9 },
    "g":   { "jawOpen": 0.4, "mouthFunnel": 0.0, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.9 },
    "ŋ":   { "jawOpen": 0.4, "mouthFunnel": 0.0, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.9 },
    "f":   { "jawOpen": 0.2, "mouthFunnel": 0.5, "mouthPucker": 0.2, "tongue_out": 0.0, "tongue_up": 0.0 },
    "v":   { "jawOpen": 0.2, "mouthFunnel": 0.5, "mouthPucker": 0.2, "tongue_out": 0.0, "tongue_up": 0.0 },
    "θ":   { "jawOpen": 0.3, "mouthFunnel": 0.3, "mouthPucker": 0.0, "tongue_out": 0.6, "tongue_up": 0.0 },
    "ð":   { "jawOpen": 0.3, "mouthFunnel": 0.3, "mouthPucker": 0.0, "tongue_out": 0.6, "tongue_up": 0.0 },
    "s":   { "jawOpen": 0.3, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.7 },
    "z":   { "jawOpen": 0.3, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.7 },
    "ʃ":   { "jawOpen": 0.3, "mouthFunnel": 0.5, "mouthPucker": 0.5, "tongue_out": 0.0, "tongue_up": 0.5 },
    "ʒ":   { "jawOpen": 0.3, "mouthFunnel": 0.5, "mouthPucker": 0.5, "tongue_out": 0.0, "tongue_up": 0.5 },
    "h":   { "jawOpen": 0.4, "mouthFunnel": 0.0, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.0 },
    "l":   { "jawOpen": 0.3, "mouthFunnel": 0.1, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.9 },
    "r":   { "jawOpen": 0.3, "mouthFunnel": 0.4, "mouthPucker": 0.5, "tongue_out": 0.0, "tongue_up": 0.4 },
    "j":   { "jawOpen": 0.2, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.6 },
    "w":   { "jawOpen": 0.2, "mouthFunnel": 0.6, "mouthPucker": 0.8, "tongue_out": 0.0, "tongue_up": 0.0 },
    "iː":  { "jawOpen": 0.2, "mouthFunnel": 0.1, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.8 },
    "ɪ":   { "jawOpen": 0.2, "mouthFunnel": 0.1, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.7 },
    "e":   { "jawOpen": 0.3, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.6 },
    "æ":   { "jawOpen": 0.5, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.4 },
    "ʌ":   { "jawOpen": 0.4, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.5 },
    "ɒ":   { "jawOpen": 0.6, "mouthFunnel": 0.3, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.3 },
    "ɔː":  { "jawOpen": 0.6, "mouthFunnel": 0.5, "mouthPucker": 0.3, "tongue_out": 0.0, "tongue_up": 0.4 },
    "ɑː":  { "jawOpen": 0.7, "mouthFunnel": 0.3, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.3 },
    "uː":  { "jawOpen": 0.2, "mouthFunnel": 0.7, "mouthPucker": 0.8, "tongue_out": 0.0, "tongue_up": 0.2 },
    "ʊ":   { "jawOpen": 0.3, "mouthFunnel": 0.6, "mouthPucker": 0.7, "tongue_out": 0.0, "tongue_up": 0.2 },
    "ɜː":  { "jawOpen": 0.4, "mouthFunnel": 0.3, "mouthPucker": 0.2, "tongue_out": 0.0, "tongue_up": 0.5 },
    "ə":   { "jawOpen": 0.3, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.4 },
    "eɪ":  { "jawOpen": 0.4, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.7 },
    "aɪ":  { "jawOpen": 0.5, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.7 },
    "ɔɪ":  { "jawOpen": 0.5, "mouthFunnel": 0.4, "mouthPucker": 0.4, "tongue_out": 0.0, "tongue_up": 0.6 },
    "aʊ":  { "jawOpen": 0.6, "mouthFunnel": 0.5, "mouthPucker": 0.6, "tongue_out": 0.0, "tongue_up": 0.4 },
    "əʊ":  { "jawOpen": 0.5, "mouthFunnel": 0.5, "mouthPucker": 0.5, "tongue_out": 0.0, "tongue_up": 0.5 },
    "ɪə":  { "jawOpen": 0.4, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.7 },
    "eə":  { "jawOpen": 0.5, "mouthFunnel": 0.2, "mouthPucker": 0.0, "tongue_out": 0.0, "tongue_up": 0.6 },
    "ʊə":  { "jawOpen": 0.5, "mouthFunnel": 0.5, "mouthPucker": 0.4, "tongue_out": 0.0, "tongue_up": 0.5 }
}

import json

def generate_phonemes(text: str, duration: numbers.Number):
    """
    Convert text into individual IPA phonemes and generate blendData.
    Ensures total duration does not exceed the specified duration.
    """
    # 1️⃣ phonemize text
    ipa_str = phonemize(
        text,
        language='en-us',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=False
    )

    # 2️⃣ remove punctuation
    ipa_str = re.sub(r'[.,!?;:]', '', ipa_str)

    # 3️⃣ split words
    words = ipa_str.split()

    # 4️⃣ multi-char phoneme list
    multi_char_phonemes = [
        "tʃ", "dʒ", "aɪ", "oʊ", "eɪ", "ɔː", "ɜː", "ʊə", "əʊ", "ɪə",
        "ɑː", "æ", "ɛ", "ɪ", "iː", "ɒ", "ʌ", "ʊ", "uː", "ɔɪ", "aʊ",
        "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z",
        "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "r", "j", "w"
    ]

    # 5️⃣ split into individual phonemes
    phonemes = []
    for word in words:
        i = 0
        while i < len(word):
            matched = False
            for m in sorted(multi_char_phonemes, key=len, reverse=True):  # match longest first
                if word[i:i + len(m)] == m:
                    phonemes.append(m)
                    i += len(m)
                    matched = True
                    break
            if not matched:
                phonemes.append(word[i])
                i += 1

    # 6️⃣ calculate step to fit total duration
    if len(phonemes) == 0:
        return []
    step = duration / len(phonemes)

    # 7️⃣ generate blendData
    blend_data = []
    current_time = 0.0

    for ph in phonemes:
        params = phenome_map.get(ph, {})  # get facial params
        datum = {"time": round(current_time, 2), "phoneme": ph}
        datum.update(params)
        blend_data.append(datum)
        current_time += step

        # ensure we don't exceed total duration
        if current_time > duration:
            break

    return blend_data


@app.route("/phonemes", methods=["POST"])
def phonemes():
    data = request.json
    text = data.get("text", "")
    result, elapsed = generate_phonemes(text)
    return jsonify({
        "phonemes": result,
        "time_taken_seconds": round(elapsed, 2)
    })


if __name__ == "__main__":
    app.run(port=3002, debug=True)
