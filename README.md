# Multilingual Help Desk (CSC 645 Azure AI Speech Project)

An international student speaks in their own language. The app transcribes the audio
(Azure AI Speech), translates it to English for staff (Azure AI Translator), analyzes it
(Azure AI Language: key phrases, entities, sentiment), and translates a reply back into
the student's language.

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

    curl -X POST -F "audio=@me_zh2.m4a" -F "language=zh-CN" http://127.0.0.1:8000/process
    curl -X POST -F "audio=@test.wav" -F "language=en-US" http://127.0.0.1:8000/transcribe
    curl -X POST -H "Content-Type: application/json" -d '{"text":"I need help renewing my visa before December."}' http://127.0.0.1:8000/analyze
    python lid_test.py test_es.wav

## Endpoints
- POST /transcribe (form: audio, language): transcript, confidence, duration, word timings
- POST /analyze (JSON: text, language): key phrases, entities, sentiment
- POST /process (form: audio, language): transcription, English translation, analysis, staff summary, reply in the student's language, per-step timings
