import os, sys
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()
cfg = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"),
                             region=os.getenv("AZURE_SPEECH_REGION"))
auto = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
    languages=["en-US", "zh-CN", "es-ES", "ko-KR"])
audio = speechsdk.audio.AudioConfig(filename=sys.argv[1])
rec = speechsdk.SpeechRecognizer(speech_config=cfg,
                                 auto_detect_source_language_config=auto,
                                 audio_config=audio)
res = rec.recognize_once()
print("Result:", res.reason)
if res.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Detected language:", speechsdk.AutoDetectSourceLanguageResult(res).language)
    print("Text:", res.text)
elif res.reason == speechsdk.ResultReason.Canceled:
    print("Error:", res.cancellation_details.error_details)
