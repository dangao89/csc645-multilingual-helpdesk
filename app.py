# app.py — Multilingual Help Desk (B: /transcribe, C1: /analyze, C2: /process, D2-D3: TTS + voices, E: web page, G: Application Insights, H: confidence-aware retry + SQLite log + /stats)

# Part G2: Application Insights MUST be initialized first (Pitfall 1)
from dotenv import load_dotenv
load_dotenv()  # load .env before everything else

from telemetry import init_telemetry
init_telemetry()  # initialize before Flask or any Azure SDK imports

import os
import json
import shutil
import subprocess
import tempfile
import threading
import time
import wave
import base64
import sqlite3                     # Part H item 3: local log database (built into Python)
import uuid
from datetime import datetime, timezone

import requests

from flask import Flask, request, jsonify, send_from_directory
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from opentelemetry import metrics      # Part G3: custom metrics
from opentelemetry import trace        # Part G4: pipeline event on the request span
import traceback                       # Part G4: safe stack trace for exceptions
from contextlib import contextmanager  # Part G5: stage_span helper
from opentelemetry.trace import Status, StatusCode   # Part G5: mark a failed stage span

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

# ---------------- Part G3: custom metrics (created once, at module level) ----------------
# Get the meter — do this once at module level, outside any function
meter = metrics.get_meter("memo-analyzer")

# Part G4: Get a tracer — at module level (used for the stage spans in G5)
tracer = trace.get_tracer("memo-analyzer")

# Create metric instruments — also at module level
stt_confidence_gauge    = meter.create_gauge("stt_confidence")
stt_duration_gauge      = meter.create_gauge("stt_duration_seconds")
stt_word_count_gauge    = meter.create_gauge("stt_word_count")
entity_count_gauge      = meter.create_gauge("language_entity_count")
keyphrase_count_gauge   = meter.create_gauge("language_keyphrase_count")
sentiment_gauge         = meter.create_gauge("language_sentiment")
tts_char_count_gauge    = meter.create_gauge("tts_char_count")
stage_stt_hist          = meter.create_histogram("stage_stt_ms")
stage_language_hist     = meter.create_histogram("stage_language_ms")
stage_tts_hist          = meter.create_histogram("stage_tts_ms")
# Our domain metric (product brief): translation latency per language
translation_hist        = meter.create_histogram("translation_latency_ms")   # name as in the product brief


def timed_stage(fn, *args, **kwargs):
    """Run fn(*args) and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


@contextmanager
def stage_span(name):
    """Part G5: same as  with tracer.start_as_current_span(name) as span:  but if the
    stage fails, only the error TYPE is recorded (raw Azure SDK messages could echo
    patient data) and the span is marked as failed in the waterfall."""
    with tracer.start_as_current_span(name, record_exception=False,
                                      set_status_on_exception=False) as span:
        try:
            yield span
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            if not isinstance(e, AudioError):      # a bad upload is the user's problem, not a failure
                span.set_status(Status(StatusCode.ERROR, f"{name} failed"))
            raise


def metric_language(locale):
    """Keep the 'language' label a small, known set (cardinality): unknown -> 'other'."""
    known = set(VOICES) | set(ALTERNATE_LOCALE) | set(ALTERNATE_LOCALE.values())
    return locale if locale in known else "other"


def emit_pipeline_metrics(stt_result, language_result, tts_result,
                          stage_timings, audio_format, requested_language):
    """Call this at the end of /process after all three stages complete."""
    # language = what the patient chose (same label as translation_ms);
    # recognition_language = the locale that won after the Part H retry (e.g. es-MX)
    attrs = {"audio_format": audio_format,
             "language": metric_language(requested_language),
             "recognition_language": metric_language(stt_result["recognition_language"])}

    # STT metrics
    if stt_result["confidence"] is not None:   # can be None if Azure gave no score
        stt_confidence_gauge.set(stt_result["confidence"], attrs)
    stt_duration_gauge.set(stt_result["duration_seconds"], attrs)
    # Word count: Azure's own word list (also works for Chinese/Japanese, which
    # have no spaces); falls back to the assignment's split() if it is empty.
    word_count = len(stt_result["words"]) or len(stt_result["transcript"].split())
    stt_word_count_gauge.set(word_count, attrs)

    # Language metrics
    entity_count_gauge.set(len(language_result["entities"]), attrs)
    keyphrase_count_gauge.set(len(language_result["key_phrases"]), attrs)
    # Azure can also return "mixed" (part positive, part negative): counted as 0,
    # and the label is attached so mixed and neutral can still be told apart.
    label = language_result["sentiment"]["label"]
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "mixed": 0.0}
    sentiment_gauge.set(
        sentiment_map.get(label, 0.0),
        {**attrs, "sentiment_label": label if label in sentiment_map else "other"},
    )

    # TTS metrics
    tts_char_count_gauge.set(tts_result["char_count"], attrs)

    # Per-stage latency histograms
    stage_stt_hist.record(stage_timings["stt_ms"], attrs)
    stage_language_hist.record(stage_timings["language_ms"], attrs)
    stage_tts_hist.record(stage_timings["tts_ms"], attrs)


def emit_pipeline_event(stt_result, lang_result, audio_format,
                        success=True, error_stage=None, error_msg=None,
                        requested_language=None, exc=None, span=None):
    """Part G4: record pipeline_completed / pipeline_error as attributes on the
    current request span (shows up as customDimensions in the requests table).
    error_msg must be a short fixed text written by us (never raw SDK/Translator
    text, which could echo patient data). exc = the original exception, passed
    only for real server/service failures (5xx); 4xx get attributes only.
    span = the request span (Part G5 passes it in, because inside pipeline.process
    the "current" span is no longer the request)."""
    try:
        span = span or trace.get_current_span()
        # language = what the patient chose (same as the G3 metrics label)
        lang = metric_language(requested_language or (stt_result or {}).get("language"))
        if success:
            span.set_attribute("event.name", "pipeline_completed")
            if stt_result["confidence"] is not None:
                span.set_attribute("stt.confidence", stt_result["confidence"])
            span.set_attribute("stt.language", lang)
            # Part H retry: the locale that was finally used (e.g. es-MX)
            span.set_attribute("stt.recognition_language",
                               metric_language(stt_result["recognition_language"]))
            span.set_attribute("entities.count", len(lang_result["entities"]))
            span.set_attribute("sentiment", lang_result["sentiment"]["label"])
            span.set_attribute("audio.format", audio_format)
        else:
            span.set_attribute("event.name", "pipeline_error")
            span.set_attribute("error.stage", error_stage)
            span.set_attribute("error.message", (error_msg or "")[:300])
            span.set_attribute("stt.language", lang)       # which language failed
            span.set_attribute("audio.format", audio_format)
            if exc is not None:
                # keep the real exception type + stack trace, but replace its raw
                # message with our safe text
                span.set_attribute("error.type", type(exc).__name__)
                # (the default stack trace ends with the raw message, so rebuild it)
                safe_trace = "".join(traceback.format_tb(exc.__traceback__)) + \
                    f"{type(exc).__name__}: {error_msg or ''}"
                span.record_exception(exc, attributes={"exception.message": error_msg or "",
                                                       "exception.stacktrace": safe_trace})
    except Exception as e:
        # Monitoring must never break the patient's request
        print("WARNING: could not emit pipeline event:", e)


def emit_translation_metric(ms, language):
    """Domain metric (product brief): translation latency per language = time to
    translate the request to English PLUS the reply back, one value per /process call,
    tagged by the patient's language. English needs no translation -> 0 ms (baseline)."""
    try:
        translation_hist.record(ms, {"language": metric_language(language)})
    except Exception as e:
        print("WARNING: could not record translation metric:", e)


ALLOWED_EXT = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".aac"}
TICKS_PER_SECOND = 10_000_000  # SDK time unit is 100 ns
LOW_WORD_CONFIDENCE = 0.70     # Part H item 1: flag words below this
RETRY_CONFIDENCE = 0.85        # Part H item 2: retry when overall confidence is below this

# Part H item 2: "a different language model hint" = another regional version of the
# same language. Korean and Japanese have only one locale in Azure, so they are not retried.
ALTERNATE_LOCALE = {
    "en-US": "en-GB", "en-GB": "en-US",
    "es-ES": "es-MX", "es-MX": "es-ES",
    "zh-CN": "zh-TW", "zh-TW": "zh-CN",
}


# ---------------- Part H item 3: SQLite log of confidence scores and retry decisions ----------------
# Locally the database file sits next to app.py.
# On Azure App Service we use /tmp (the instance's own local disk). We do NOT use /home:
# /home is a network share, and Microsoft says SQLite is not supported there because file
# locking is unreliable. Trade-off: the /tmp log is cleared when the app restarts.
if os.environ.get("WEBSITE_SITE_NAME"):          # this variable only exists on App Service
    DB_PATH = "/tmp/confidence_log.db"
else:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confidence_log.db")

# A new id every time the server starts, so /stats can report "this session" separately.
# (startup.sh runs gunicorn with its default of ONE worker, so one server run = one session.)
SESSION_ID = uuid.uuid4().hex


def db_connect():
    # wait up to 10 s if another request is writing at the same moment
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    """Create the database file, table and index if they do not exist yet."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db_connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS confidence_log (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at            TEXT,     -- UTC time, ISO format
                session_id            TEXT,     -- which server run this came from
                endpoint              TEXT,     -- /process or /transcribe
                requested_locale      TEXT,     -- language the user picked, e.g. es-ES
                used_locale           TEXT,     -- locale whose result was kept
                original_confidence   REAL,     -- first try, overall score (unrounded)
                retry_attempted       INTEGER,  -- 1 = re-submitted with another locale
                retry_locale          TEXT,
                retry_confidence      REAL,     -- retry, overall score (unrounded)
                retry_used            INTEGER,  -- 1 = the retry result was kept
                final_confidence      REAL,     -- overall score of the kept result (unrounded)
                retry_reason          TEXT,     -- why we retried / kept / switched
                word_count            INTEGER,  -- words in the kept result
                word_conf_count       INTEGER,  -- of those, words that have a score
                word_conf_avg         REAL,
                word_conf_min         REAL,
                low_confidence_words  INTEGER,  -- words below 0.70 in the kept result
                attempts_json         TEXT,     -- EVERY attempt: locale, score and each word's score
                duration_seconds      REAL,
                first_stt_ms          INTEGER,
                retry_stt_ms          INTEGER
            )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_confidence_log_session ON confidence_log(session_id)")


def log_transcription(result, endpoint):
    """Write one row per transcription. Logging must never break the request,
    so any database error is printed and ignored. Returns the new row id or None."""
    attempts = result.pop("_attempts", [])      # internal only: not sent to the browser
    r = result["retry"]
    scores = [w["confidence"] for w in result["words"] if w["confidence"] is not None]
    raw = lambda n: attempts[n]["confidence"] if len(attempts) > n else None
    try:
        with db_connect() as db:
            cur = db.execute(
                """INSERT INTO confidence_log (created_at, session_id, endpoint, requested_locale,
                   used_locale, original_confidence, retry_attempted, retry_locale, retry_confidence,
                   retry_used, final_confidence, retry_reason, word_count, word_conf_count,
                   word_conf_avg, word_conf_min, low_confidence_words, attempts_json,
                   duration_seconds, first_stt_ms, retry_stt_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    SESSION_ID, endpoint, r["original_locale"], r["used_locale"],
                    raw(0), int(r["attempted"]), r["retry_locale"], raw(1),
                    int(r["used_locale"] != r["original_locale"]),
                    result["confidence_raw"], r.get("reason"),
                    len(result["words"]), len(scores),
                    sum(scores) / len(scores) if scores else None,
                    min(scores) if scores else None,
                    len(result["low_confidence_words"]),
                    json.dumps(attempts, ensure_ascii=False),
                    result["duration_seconds"], r["first_stt_ms"], r["retry_stt_ms"],
                ),
            )
            return cur.lastrowid
    except sqlite3.Error as e:
        print("WARNING: could not write to the SQLite log:", e)
        return None


try:
    init_db()
except sqlite3.Error as e:          # never stop the app because of the log
    print("WARNING: could not create the SQLite log:", e)


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
            # Part H item 1: use the WORD's own score only. If Azure gives none,
            # keep None (do not borrow the sentence score).
            conf = w.get("Confidence")
            words.append({
                "word": w.get("Word"),
                "offset": round(w.get("Offset", 0) / TICKS_PER_SECOND, 2),
                "duration": round(w.get("Duration", 0) / TICKS_PER_SECOND, 2),
                "confidence": conf,
                # Part H item 1: flag words the recognizer is unsure about
                "low_confidence": conf is not None and conf < LOW_WORD_CONFIDENCE,
            })

    with wave.open(wav_path, "rb") as w:
        duration = w.getnframes() / float(w.getframerate())

    return {
        "transcript": " ".join(t for t in texts if t),
        "language": language,
        "duration_seconds": round(duration, 2),
        "confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        # Part H item 2: unrounded mean of the segment scores, used for the 0.85 decision
        # (0.846 must NOT be rounded up to 0.85 and skip the retry)
        "confidence_raw": sum(confidences) / len(confidences) if confidences else None,
        "words": words,
        # Part H item 1: flagged words with their position in "words" and their score,
        # so a word that appears twice is never confused with its other copy.
        "low_confidence_words": [
            {"index": i, "word": w["word"], "confidence": w["confidence"]}
            for i, w in enumerate(words) if w["low_confidence"]
        ],
        "word_confidence_available": any(w["confidence"] is not None for w in words),
    }


@app.route("/")
def home():
    """Part E: open http://localhost:8000 to get the web page (index.html in this folder)."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


class AudioError(Exception):
    """Bad upload: carries the HTTP status code to return (400 or 415)."""
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def transcribe_upload(file, language, endpoint="/transcribe"):
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
            result = transcribe_with_retry(wav_path, language)   # Part H item 2
            result["log_id"] = log_transcription(result, endpoint)  # Part H item 3
            return result
        except RuntimeError as e:
            msg = str(e)
            if "format" in msg.lower() or "header" in msg.lower():
                raise AudioError("Unsupported audio format: " + msg, 415)
            raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # delete temp file


def summarize_attempt(locale, res):
    """Part H item 3: what we keep about each attempt (also the one we throw away)."""
    return {"locale": locale, "confidence": res["confidence_raw"], "transcript": res["transcript"],
            "words": [{"word": w["word"], "confidence": w["confidence"]} for w in res["words"]]}


def transcribe_with_retry(wav_path, language):
    """Part H item 2: transcribe once; if overall confidence is below 0.85, re-submit the
    same audio with the alternate locale and keep whichever result is better."""
    t = time.time()
    first = transcribe_wav(wav_path, language)
    first_ms = int((time.time() - t) * 1000)
    attempts = [summarize_attempt(language, first)]   # Part H item 3: logged to SQLite
    # Overall confidence = mean of the segment scores. Compare the UNROUNDED value;
    # no speech recognized counts as 0.
    first_conf = first["confidence_raw"] if first["confidence_raw"] is not None else 0.0
    alt = ALTERNATE_LOCALE.get(language)
    retry = {
        "threshold": RETRY_CONFIDENCE,
        "original_locale": language,
        "original_confidence": first["confidence"],
        "attempted": False,
        "retry_locale": None,
        "retry_confidence": None,
        "used_locale": language,
        "first_stt_ms": first_ms,
        "retry_stt_ms": None,
    }

    def keep_first(reason):
        retry["reason"] = reason
        return {**first, "recognition_language": language, "retry": retry, "_attempts": attempts}

    if first_conf >= RETRY_CONFIDENCE:
        return keep_first(f"confidence {first['confidence']} is at or above {RETRY_CONFIDENCE}: no retry needed")
    if alt is None:
        return keep_first(f"confidence {first['confidence']} is below {RETRY_CONFIDENCE}, but {language} has no alternate locale")

    retry["attempted"] = True
    retry["retry_locale"] = alt
    t = time.time()
    try:
        second = transcribe_wav(wav_path, alt)
    except Exception as e:   # the retry is optional: any failure keeps the first result
        retry["retry_stt_ms"] = int((time.time() - t) * 1000)
        return keep_first(f"retry with {alt} failed ({type(e).__name__}: {str(e)[:100]}): kept {language} result")
    retry["retry_stt_ms"] = int((time.time() - t) * 1000)
    retry["retry_confidence"] = second["confidence"]
    attempts.append(summarize_attempt(alt, second))
    second_conf = second["confidence_raw"] if second["confidence_raw"] is not None else 0.0

    if not second["transcript"]:
        return keep_first(f"{alt} recognized no speech: kept {language} result")
    if not first["transcript"] or second_conf > first_conf:
        why = (f"{language} recognized no speech" if not first["transcript"]
               else f"{alt} scored higher ({second['confidence']} > {first['confidence']})")
        retry["used_locale"] = alt
        retry["reason"] = f"{why}: using {alt} result"
        return {**second, "recognition_language": alt, "retry": retry, "_attempts": attempts}
    return keep_first(f"{alt} did not score higher ({second['confidence']} <= {first['confidence']}): kept {language} result")


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


# ---------------- Part D: Text-to-Speech ----------------

# D3: three neural voices per language, each with a different persona.
# The first voice in each list is the default.
VOICES = {
    "en-US": [("en-US-JennyNeural", "friendly"), ("en-US-GuyNeural", "formal"), ("en-US-AriaNeural", "energetic")],
    "zh-CN": [("zh-CN-XiaoxiaoNeural", "warm"), ("zh-CN-YunyangNeural", "formal"), ("zh-CN-YunxiNeural", "lively")],
    "es-ES": [("es-ES-ElviraNeural", "friendly"), ("es-ES-AlvaroNeural", "formal"), ("es-ES-AbrilNeural", "energetic")],
    "ko-KR": [("ko-KR-SunHiNeural", "friendly"), ("ko-KR-InJoonNeural", "formal"), ("ko-KR-BongJinNeural", "casual")],
    "ja-JP": [("ja-JP-NanamiNeural", "friendly"), ("ja-JP-KeitaNeural", "formal"), ("ja-JP-AoiNeural", "energetic")],
}
# Any other language: multilingual voices that can read text in many languages.
MULTILINGUAL_VOICES = [("en-US-AvaMultilingualNeural", "friendly"),
                       ("en-US-AndrewMultilingualNeural", "formal"),
                       ("en-US-EmmaMultilingualNeural", "energetic")]


def voice_options(language):
    """D3: the three (voice, persona) choices for this language."""
    return VOICES.get(language, MULTILINGUAL_VOICES)


def pick_voice(language, choice):
    """D3: use the voice the user chose if it belongs to this language, else the default."""
    names = [v for v, _ in voice_options(language)]
    return choice if choice in names else names[0]


@app.route("/voices")
def voices():
    """D3: GET /voices?language=zh-CN -> the voices the frontend dropdown can show."""
    language = request.args.get("language", "en-US")
    return jsonify({
        "language": language,
        "voices": [{"name": v, "persona": p} for v, p in voice_options(language)],
    })


def synthesize(text, voice):
    """D2: Azure neural TTS -> MP3 bytes (Audio16Khz32KBitRateMonoMp3)."""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = result.cancellation_details
        raise RuntimeError(f"TTS failed: {details.reason} {details.error_details}")
    return result.audio_data


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
    # Part G5: remember the request span that Azure created automatically (G2), so the
    # G4 event stays on the request (requests table) even inside pipeline.process below
    request_span = trace.get_current_span()

    audio_file = request.files.get("audio")
    audio_format = os.path.splitext((audio_file.filename if audio_file else "") or "")[1].lstrip(".").lower() or "unknown"

    # Part G5: parent span for the whole pipeline; the stage spans nest inside it
    with tracer.start_as_current_span("pipeline.process", record_exception=False,
                                      set_status_on_exception=False) as root_span:
        root_span.set_attribute("audio.format", audio_format)
        try:
            result = run_pipeline(audio_file, audio_format, request_span)
        except Exception as e:
            # something unexpected escaped run_pipeline (ChatGPT G5 review):
            # mark the root span failed (type only, no raw message), then let Flask return 500
            root_span.set_attribute("error.type", type(e).__name__)
            root_span.set_status(Status(StatusCode.ERROR, "pipeline failed"))
            raise
        status = result[1] if isinstance(result, tuple) else 200
        if status >= 500:
            root_span.set_status(Status(StatusCode.ERROR, "pipeline failed"))
        return result


def run_pipeline(audio_file, audio_format, request_span):
    """The /process pipeline (Parts C-H). Each stage runs inside its own G5 span."""
    language = request.form.get("language", "en-US")  # the patient's language
    tr_code = to_translator_code(language)
    timings = {}

    # 1-2. audio -> transcript (patient's language)  [G3: timed stage 1 = STT] [G5: span]
    try:
        with stage_span("stage.speech_to_text") as stt_span:
            stt, stt_ms = timed_stage(transcribe_upload, audio_file, language, endpoint="/process")
            if stt["confidence"] is not None:
                stt_span.set_attribute("stt.confidence", stt["confidence"])
            stt_span.set_attribute("stt.word_count", len(stt["words"]) or len(stt["transcript"].split()))
            stt_span.set_attribute("duration_ms", stt_ms)
    except AudioError as e:
        # Part G4: bad upload (missing file, unsupported format, too large)
        # (4xx -> attributes only, no exception recorded)
        # telemetry gets a fixed text only: str(e) can contain a raw Speech SDK message
        # (ChatGPT G5 review); the patient still sees the full message in the response
        emit_pipeline_event(None, None, audio_format, success=False, error_stage="upload",
                            error_msg=f"Bad upload (HTTP {e.status})", requested_language=language,
                            span=request_span)
        return jsonify({"error": str(e)}), e.status
    except RuntimeError as e:
        emit_pipeline_event(None, None, audio_format, success=False, error_stage="stt",
                            error_msg="Speech service error", requested_language=language, exc=e,
                            span=request_span)
        return jsonify({"error": "Speech service error: " + str(e)}), 502
    except Exception as e:
        # anything unexpected in upload/STT (ChatGPT G4 review)
        emit_pipeline_event(None, None, audio_format, success=False, error_stage="stt",
                            error_msg="Unexpected STT error", requested_language=language, exc=e,
                            span=request_span)
        return jsonify({"error": "Speech-to-text failed: " + str(e)[:300]}), 502
    timings["stt_ms"] = int(stt_ms)   # includes the retry, if any

    if not stt["transcript"]:
        emit_pipeline_event(stt, None, audio_format, success=False, error_stage="stt",
                            error_msg="No speech recognized in the audio.", requested_language=language,
                            span=request_span)
        return jsonify({"error": "No speech recognized in the audio.", "transcription": stt}), 422

    stage = "translate_to_en"   # Part G4: which step we are in, for error.stage
    try:
        # 3. translate to English for staff (skip if already English)
        t = time.time()
        # Part H item 2: translate from the locale that was actually used (zh-TW gives
        # Traditional characters, so the source code must be zh-Hant, not zh-Hans)
        src_code = to_translator_code(stt["recognition_language"])
        with stage_span("stage.translate_to_en") as tr_span:      # G5 (our extra stage)
            text_en = stt["transcript"] if src_code == "en" else translate(stt["transcript"], "en", src_code)
            timings["translate_to_en_ms"] = int((time.time() - t) * 1000)
            tr_span.set_attribute("translation.skipped", src_code == "en")
            tr_span.set_attribute("duration_ms", timings["translate_to_en_ms"])

        # 4. Language analysis on the English text  [Part G3: timed stage 2 = Language]
        stage = "language"
        with stage_span("stage.language_analysis") as lang_span:   # G5
            analysis, language_ms = timed_stage(analyze_text, text_en, "en")
            lang_span.set_attribute("entity_count", len(analysis["entities"]))
            lang_span.set_attribute("sentiment", analysis["sentiment"]["label"])
            lang_span.set_attribute("duration_ms", language_ms)
        timings["analyze_ms"] = int(language_ms)

        # 5. summary for staff + reply translated back into the patient's language
        summary_en = build_summary(analysis)
        reply_en = build_reply_en(analysis)
        stage = "translate_back"
        t = time.time()
        with stage_span("stage.translate_back") as tb_span:      # G5 (our extra stage)
            reply_text = reply_en if tr_code == "en" else translate(reply_en, tr_code, "en")
            timings["translate_back_ms"] = int((time.time() - t) * 1000)
            tb_span.set_attribute("translation.skipped", tr_code == "en")
            tb_span.set_attribute("duration_ms", timings["translate_back_ms"])
        # Domain metric: request -> English + reply -> patient language (0 for English)
        emit_translation_metric(timings["translate_to_en_ms"] + timings["translate_back_ms"], language)

        # 6. Part D2: speak the reply in the patient's language
        voice = pick_voice(language, request.form.get("voice"))  # D3: user's choice
        # [Part G3: timed stage 3 = TTS]
        stage = "tts"
        with stage_span("stage.text_to_speech") as tts_span:      # G5
            audio_bytes, tts_ms = timed_stage(synthesize, reply_text, voice)
            tts_span.set_attribute("char_count", len(reply_text))
            tts_span.set_attribute("duration_ms", tts_ms)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        timings["tts_ms"] = int(tts_ms)
    except Exception as e:
        emit_pipeline_event(stt, None, audio_format, success=False, error_stage=stage,
                            error_msg=f"{stage} failed", requested_language=language, exc=e,
                            span=request_span)
        return jsonify({"error": "Pipeline error: " + str(e)[:300], "transcription": stt}), 502

    # Part G3: all three stages succeeded -> send the custom metrics to Application Insights.
    # Monitoring must never break the patient's request, so errors are only printed.
    try:
        emit_pipeline_metrics(
            stt, analysis, {"char_count": len(reply_text)},
            {"stt_ms": stt_ms, "language_ms": language_ms, "tts_ms": tts_ms},
            audio_format, language,
        )
    except Exception as e:
        print("WARNING: could not emit pipeline metrics:", e)

    # Part G4: success event on the request span
    emit_pipeline_event(stt, analysis, audio_format, success=True, requested_language=language,
                            span=request_span)

    return jsonify({
        "language": language,
        "transcription": {
            "text": stt["transcript"],
            "confidence": stt["confidence"],
            "duration_seconds": stt["duration_seconds"],
            "words": stt["words"],                              # Part H item 1: per-word confidence + flag
            "low_confidence_words": stt["low_confidence_words"],
            "word_confidence_available": stt["word_confidence_available"],
            "low_confidence_threshold": LOW_WORD_CONFIDENCE,   # the UI reads it from here
            "recognition_language": stt["recognition_language"],  # Part H item 2
            "retry": stt["retry"],                                # Part H item 2: retry decision
            "log_id": stt["log_id"],                              # Part H item 3: row in SQLite
        },
        "translation_en": text_en,
        "analysis": analysis,
        "summary_en": summary_en,
        "reply": {
            "text_en": reply_en,
            "text": reply_text,
            "language": language,
            "voice": voice,
            "audio_base64": audio_b64,  # browser plays it as data:audio/mp3;base64,...
        },
        "timings_ms": timings,
    })


# ---------------- Part H item 4: GET /stats ----------------
# Shared SELECT used for the overall numbers and for each language.
STATS_SQL = """
    SELECT COUNT(*)                                              AS transcriptions,
           COUNT(final_confidence)                               AS with_confidence,
           AVG(final_confidence)                                 AS avg_confidence,
           MIN(final_confidence)                                 AS min_confidence,
           MAX(final_confidence)                                 AS max_confidence,
           COUNT(original_confidence)                            AS first_tries_with_confidence,
           AVG(original_confidence)                              AS avg_first_try_confidence,
           SUM(CASE WHEN original_confidence IS NULL
                     OR original_confidence < ? THEN 1 ELSE 0 END) AS below_retry_threshold,
           SUM(retry_attempted)                                  AS retries_attempted,
           SUM(retry_used)                                       AS retries_used,
           SUM(word_count)                                       AS words,
           SUM(low_confidence_words)                             AS low_confidence_words,
           SUM(word_conf_count)                                  AS words_with_confidence,
           SUM(word_conf_avg * word_conf_count) / NULLIF(SUM(word_conf_count), 0)
                                                                 AS avg_word_confidence,
           MIN(word_conf_min)                                    AS min_word_confidence,
           AVG(first_stt_ms)                                     AS avg_first_stt_ms,
           AVG(retry_stt_ms)                                     AS avg_retry_stt_ms
    FROM confidence_log
"""


def tidy(row):
    """Round the numbers so the JSON is easy to read (the database keeps full precision)."""
    out = {}
    row = dict(row)
    n, words = row["transcriptions"] or 0, row["words_with_confidence"] or 0
    row["retry_rate"] = (row["retries_attempted"] or 0) / n if n else None
    row["low_confidence_word_rate"] = (row["low_confidence_words"] or 0) / words if words else None
    for k, v in row.items():
        if isinstance(v, float):
            v = round(v) if k.endswith("_ms") else round(v, 3)
        out[k] = v if v is not None else (0 if k in ("retries_attempted", "retries_used", "words",
                                                      "low_confidence_words", "below_retry_threshold",
                                                      "words_with_confidence") else None)
    return out


@app.route("/stats")
def stats():
    """Part H item 4: aggregate confidence statistics from the SQLite log.
    Default = this session (since the server started). ?scope=all = every row in the database."""
    scope = request.args.get("scope", "session")
    if scope not in ("session", "all"):
        return jsonify({"error": "scope must be 'session' or 'all'"}), 400
    where, params = ("WHERE session_id = ?", [SESSION_ID]) if scope == "session" else ("", [])
    try:
        with db_connect() as db:
            db.row_factory = sqlite3.Row
            overall = db.execute(STATS_SQL + where, [RETRY_CONFIDENCE] + params).fetchone()
            by_lang = db.execute(
                STATS_SQL.replace("SELECT ", "SELECT requested_locale AS locale, ", 1) + where +
                " GROUP BY requested_locale ORDER BY requested_locale",
                [RETRY_CONFIDENCE] + params).fetchall()
    except sqlite3.Error as e:
        return jsonify({"error": "Could not read the SQLite log: " + str(e)}), 500

    return jsonify({
        "scope": scope,
        "session_id": SESSION_ID if scope == "session" else None,
        "thresholds": {"retry_below": RETRY_CONFIDENCE, "word_flag_below": LOW_WORD_CONFIDENCE},
        "overall": tidy(overall),
        "by_language": [tidy(r) for r in by_lang],
        "notes": [
            "avg/min/max_confidence = overall score of the result that was kept (after any retry); "
            "with_confidence is their denominator",
            "avg_first_try_confidence = before any retry; first_tries_with_confidence is its denominator",
            "below_retry_threshold also counts first tries with no score (no speech), because those are retried too",
            "avg_word_confidence is weighted by words_with_confidence",
            "avg_retry_stt_ms averages only the requests that were actually retried",
            "by_language groups by the language the user picked (requested_locale), not the locale finally used",
        ],
    })


@app.errorhandler(413)
def too_large(e):
    # Part G4: Flask rejects a >25 MB upload before /process runs, so record the event here
    if request.path == "/process":
        emit_pipeline_event(None, None, "unknown", success=False, error_stage="upload",
                            error_msg="File too large (over 25 MB)")
    return jsonify({"error": "File too large. Max 25 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
