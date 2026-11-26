"""Make ~15-second native-voice test recordings (Spanish, Japanese, Korean) with Azure TTS.
Uses the Speech key from .env (never printed). Output: native_es.mp3, native_ja.mp3, native_ko.mp3
Run: python make_test_audio.py
"""
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()
cfg = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"),
                             region=os.getenv("AZURE_SPEECH_REGION"))
cfg.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)

SAMPLES = {
    # Spanish (Spain): fever, cough, appointment
    "native_es.mp3": ("es-ES-AlvaroNeural",
        "Buenas tardes. Desde hace tres días tengo fiebre y mucha tos. "
        "Anoche casi no pude dormir y hoy me duele la garganta. "
        "¿Puedo ver a un médico esta tarde, por favor?"),
    # Japanese: stomach pain, vomiting
    "native_ja.mp3": ("ja-JP-KeitaNeural",
        "こんにちは。昨日の夜からお腹がとても痛いです。"
        "今朝、二回吐きました。熱は少しあります。"
        "今日、先生に診てもらえますか。"),
    # Korean: headache, dizziness, follow-up
    "native_ko.mp3": ("ko-KR-InJoonNeural",
        "안녕하세요. 일주일 전부터 머리가 계속 아프고 가끔 어지럽습니다. "
        "지난주에 김 선생님께 진료를 받았는데 아직 좋아지지 않았어요. "
        "다음 주 월요일에 다시 예약할 수 있을까요?"),
}

for filename, (voice, text) in SAMPLES.items():
    cfg.speech_synthesis_voice_name = voice
    # audio_config=None: keep the audio in memory (no speaker, no file handle inside the SDK),
    # then write the bytes ourselves. Writing straight to a file crashed on macOS (segfault).
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    result = synth.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        with open(filename, "wb") as f:
            f.write(result.audio_data)
        print(f"OK  {filename}  ({voice}, {len(result.audio_data)} bytes)", flush=True)
    else:
        print(f"FAILED {filename}: {result.reason}", flush=True)
    del synth
