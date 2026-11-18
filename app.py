# app.py — Multilingual Help Desk (B: /transcribe, C1: /analyze, C2: /process)
import os
import json
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import requests

from dotenv import load_dotenv
from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
TRANSLATOR_URL = "https://api.cognitive.microsofttranslator.com/translate"

text_client = TextAnalyticsClient(
    endpoint=LANGUAGE_ENDPOINT, credential=AzureKeyCredential(LANGUAGE_KEY)
)

app = Flask(__name__)
app.json.ensure_ascii = False  # show Chinese etc. as real characters
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB limit

ALLOWED_EXT = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".aac"}
TICKS_PER_SECOND = 10_000_000  # SDK time unit is 100 ns


def to_wav_16k_mono(src_path):
    """Convert non-WAV audio to 16kHz mono WAV with ffmpeg. Returns None if not possible."""
    if shutil.which("ffmpeg") is None:
        return None
    out_path = src_path + ".converted.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", out_path],
        capture_output=True,
    )
    return out_path if result.returncode == 0 else None


def is_valid_wav(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() > 0
    except Exception:
        return False


def transcribe_wav(wav_path, language):
    """Transcribe the whole file with continuous recognition."""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = language
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    speech_config.request_word_level_timestamps()

    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = threading.Event()   # Pitfall 3: wait for the callback to finish
    segments = []
    errors = []

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            segments.append(json.loads(evt.result.json))

    def on_canceled(evt):
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            errors.append(evt.cancellation_details.error_details)
        done.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(lambda evt: done.set())

    recognizer.start_continuous_recognition_async().get()
    done.wait(timeout=300)
    recognizer.stop_continuous_recognition_async().get()

    if errors:
        raise RuntimeError(errors[0])

    texts, confidences, words = [], [], []
    for seg in segments:
        best = (seg.get("NBest") or [{}])[0]
        texts.append(seg.get("DisplayText", ""))
        if "Confidence" in best:
            confidences.append(best["Confidence"])
        for w in best.get("Words", []):
            words.append({
                "word": w.get("Word"),
                "offset": round(w.get("Offset", 0) / TICKS_PER_SECOND, 2),
                "duration": round(w.get("Duration", 0) / TICKS_PER_SECOND, 2),
                "confidence": w.get("Confidence", best.get("Confidence")),
            })

    with wave.open(wav_path, "rb") as w:
        duration = w.getnframes() / float(w.getframerate())

    return {
        "transcript": " ".join(t for t in texts if t),
        "language": language,
        "duration_seconds": round(duration, 2),
        "confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "words": words,
    }


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


class AudioError(Exception):
    """Bad upload: carries the HTTP status code to return (400 or 415)."""
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def transcribe_upload(file, language):
    """Save the upload to a temp folder, convert if needed, transcribe, then delete it."""
    if file is None:
        raise AudioError("No file uploaded. Use form field 'audio'.", 400)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise AudioError(f"Unsupported audio format '{ext}'. Please upload WAV or MP3.", 415)

    tmp_dir = tempfile.mkdtemp()
    src_path = os.path.join(tmp_dir, "upload" + ext)
    file.save(src_path)
    try:
        wav_path = src_path if ext == ".wav" else to_wav_16k_mono(src_path)
        if wav_path is None or not is_valid_wav(wav_path):
            raise AudioError("Could not read this audio. Please upload a 16kHz mono WAV file.", 415)
        try:
            return transcribe_wav(wav_path, language)
        except RuntimeError as e:
            msg = str(e)
            if "format" in msg.lower() or "header" in msg.lower():
                raise AudioError("Unsupported audio format: " + msg, 415)
            raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # delete temp file


@app.route("/transcribe", methods=["POST"])
def transcribe():
    language = request.form.get("language", "en-US")  # e.g. zh-CN, es-ES
    try:
        return jsonify(transcribe_upload(request.files.get("audio"), language))
    except AudioError as e:
        return jsonify({"error": str(e)}), e.status
    except RuntimeError as e:
        return jsonify({"error": "Speech service error: " + str(e)}), 502


# ---------------- Part C: Language analysis ----------------

def to_language_code(locale):
    """Speech uses 'zh-CN', 'es-ES'; the Language service wants 'zh-hans', 'es'."""
    locale = (locale or "en").strip()
    special = {"zh-cn": "zh-hans", "zh-sg": "zh-hans", "zh-tw": "zh-hant", "zh-hk": "zh-hant"}
    return special.get(locale.lower(), locale.split("-")[0].lower())


def analyze_text(text, locale="en"):
    """Key phrases, named entities, sentiment (+ optional linked entities)."""
    lang = to_language_code(locale)
    docs = [text]

    kp = text_client.extract_key_phrases(docs, language=lang)[0]
    ner = text_client.recognize_entities(docs, language=lang)[0]
    sent = text_client.analyze_sentiment(docs, language=lang)[0]
    for r in (kp, ner, sent):
        if r.is_error:
            raise RuntimeError(r.error.message)

    # Linked entities are optional: Microsoft stopped onboarding new customers in 2025.
    try:
        le = text_client.recognize_linked_entities(docs, language=lang)[0]
        if le.is_error:
            linked = {"available": False, "reason": le.error.message}
        else:
            linked = {"available": True, "items": [
                {"name": e.name, "url": e.url, "source": e.data_source} for e in le.entities
            ]}
    except Exception as e:
        linked = {"available": False, "reason": str(e)[:200]}

    return {
        "text": text,
        "language": lang,
        "key_phrases": list(kp.key_phrases),
        "entities": [
            {"text": e.text, "category": e.category, "subcategory": e.subcategory,
             "confidence": round(e.confidence_score, 2)}
            for e in ner.entities
        ],
        "sentiment": {
            "label": sent.sentiment,
            "scores": {
                "positive": round(sent.confidence_scores.positive, 2),
                "neutral": round(sent.confidence_scores.neutral, 2),
                "negative": round(sent.confidence_scores.negative, 2),
            },
        },
        "linked_entities": linked,
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Send JSON like {\"text\": \"...\"}."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long. Max 5000 characters."}), 400
    try:
        return jsonify(analyze_text(text, data.get("language", "en")))
    except Exception as e:
        return jsonify({"error": "Language service error: " + str(e)[:300]}), 502


# ---------------- Part C2: full pipeline ----------------

def to_translator_code(locale):
    """Speech uses 'zh-CN'; Translator wants 'zh-Hans', 'es', 'ko'..."""
    locale = (locale or "en").strip()
    special = {"zh-cn": "zh-Hans", "zh-sg": "zh-Hans", "zh-tw": "zh-Hant", "zh-hk": "zh-Hant"}
    return special.get(locale.lower(), locale.split("-")[0].lower())


def translate(text, to_lang, from_lang=None):
    """Azure AI Translator (REST). Returns the translated text."""
    params = {"api-version": "3.0", "to": to_lang}
    if from_lang:
        params["from"] = from_lang
    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json",
    }
    r = requests.post(TRANSLATOR_URL, params=params, headers=headers, json=[{"text": text}], timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Translator error {r.status_code}: {r.text[:200]}")
    return r.json()[0]["translations"][0]["text"]


def build_summary(analysis):
    """D1: key phrase count + topics, sentiment, and entity types found (English, for staff)."""
    phrases = analysis["key_phrases"]
    counts = {}
    for e in analysis["entities"]:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    parts = [f"The request mentions {len(phrases)} key topics"]
    parts[0] += (": " + ", ".join(phrases[:5]) + ".") if phrases else "."
    parts.append(f"The overall tone is {analysis['sentiment']['label']}.")
    if counts:
        parts.append("I detected " + ", ".join(f"{n} {c}" for c, n in counts.items()) + ".")
    return " ".join(parts)


def build_reply_en(analysis):
    """Short acknowledgement for the patient (translated into their language later).
    Key phrases go to staff in summary_en; listing them here sounded awkward when translated."""
    dates = [e["text"] for e in analysis["entities"] if e["category"] == "DateTime"]
    reply = "Thank you. We have received your request."
    if dates:
        reply += f" We noted the time you mentioned: {dates[0]}."
    return reply + " A staff member will help you shortly."

@app.route("/process", methods=["POST"])
def process():
    language = request.form.get("language", "en-US")  # the patient's language
    tr_code = to_translator_code(language)
    timings = {}

    # 1-2. audio -> transcript (patient's language)
    t = time.time()
    try:
        stt = transcribe_upload(request.files.get("audio"), language)
    except AudioError as e:
        return jsonify({"error": str(e)}), e.status
    except RuntimeError as e:
        return jsonify({"error": "Speech service error: " + str(e)}), 502
    timings["stt_ms"] = int((time.time() - t) * 1000)

    if not stt["transcript"]:
        return jsonify({"error": "No speech recognized in the audio.", "transcription": stt}), 422

    try:
        # 3. translate to English for staff (skip if already English)
        t = time.time()
        text_en = stt["transcript"] if tr_code == "en" else translate(stt["transcript"], "en", tr_code)
        timings["translate_to_en_ms"] = int((time.time() - t) * 1000)

        # 4. Language analysis on the English text
        t = time.time()
        analysis = analyze_text(text_en, "en")
        timings["analyze_ms"] = int((time.time() - t) * 1000)

        # 5. summary for staff + reply translated back into the patient's language
        summary_en = build_summary(analysis)
        reply_en = build_reply_en(analysis)
        t = time.time()
        reply_text = reply_en if tr_code == "en" else translate(reply_en, tr_code, "en")
        timings["translate_back_ms"] = int((time.time() - t) * 1000)
    except Exception as e:
        return jsonify({"error": "Pipeline error: " + str(e)[:300], "transcription": stt}), 502

    return jsonify({
        "language": language,
        "transcription": {
            "text": stt["transcript"],
            "confidence": stt["confidence"],
            "duration_seconds": stt["duration_seconds"],
        },
        "translation_en": text_en,
        "analysis": analysis,
        "summary_en": summary_en,
        "reply": {
            "text_en": reply_en,
            "text": reply_text,
            "language": language,
            "audio_base64": None,  # Part D: TTS goes here
        },
        "timings_ms": timings,
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 25 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
