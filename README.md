# Multilingual Help Desk (CSC 645 Azure AI Speech Project)

A patient or family member with limited English speaks in their own language at a hospital
or clinic, at the front desk or with a doctor. The app transcribes the audio (Azure AI Speech),
translates it to English for staff (Azure AI Translator), analyzes it (Azure AI Language:
key phrases, entities, sentiment), and translates a reply back into the patient's language.
It supports routine communication and does not replace a certified medical interpreter.

## Status (checkpoint)
- Part A: Azure resources provisioned in swedencentral: Speech, Language and Translator (F0), Log Analytics + Application Insights
- Part B: POST /transcribe, continuous recognition, WAV/MP3/OGG/WebM/M4A, 415 for unsupported audio
- Part C: POST /analyze and POST /process (full pipeline), tested on localhost
- Track 1 test: lid_test.py shows automatic language identification works on our F0 resource

## Run locally
1. pip install -r requirements.txt   (ffmpeg is also needed for non-WAV audio)
2. Create a .env file (not in this repo) with:
   AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, AZURE_LANGUAGE_KEY, AZURE_LANGUAGE_ENDPOINT,
   AZURE_TRANSLATOR_KEY, AZURE_TRANSLATOR_REGION
3. python app.py   (runs on http://127.0.0.1:8000)

## Try it

    curl -X POST -F "audio=@hospital_zh.m4a" -F "language=zh-CN" http://127.0.0.1:8000/process
    curl -X POST -F "audio=@test.mp3" -F "language=en-US" http://127.0.0.1:8000/transcribe
    curl -X POST -H "Content-Type: application/json" -d '{"text":"I have had a fever since last night and I took two ibuprofen."}' http://127.0.0.1:8000/analyze

## Test recordings (recorded myself, results in out_*.json)

| File | Language | Confidence | Result |
|---|---|---|---|
| hospital_zh.m4a | zh-CN | 0.90 | Correct (fever 39°C, ibuprofen) |
| hospital_en.m4a | en-US | 0.79 | Correct (reschedule, December 3rd) |
| hospital_ko.m4a | ko-KR | 0.83 | Partly correct ("stomach" misheard) |
| hospital_es.m4a | es-ES | 0.56 | Partial (read by a non-native speaker) |
| hospital_ja.m4a | ja-JP | 0.55 | Partial (read by a non-native speaker) |

The two lowest-confidence recordings (0.55 and 0.56) had the most errors, which motivates the Part H confidence-aware retry.
test.wav/mp3/ogg/webm/m4a are hospital_en converted to each format for the Part B format tests.

## Endpoints
- POST /transcribe (form: audio, language): transcript, confidence, duration, word timings
- POST /analyze (JSON: text, language): key phrases, entities, sentiment
- POST /process (form: audio, language): transcription, English translation, analysis, staff summary, reply in the patient's language, per-step timings
