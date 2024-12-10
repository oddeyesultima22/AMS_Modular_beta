import whisper

def transcribe_audio(audio_file, model_name="tiny.en"):
    try:
        model = whisper.load_model(model_name, device="cpu")  # Ensures FP32 is used.
        return model.transcribe(audio_file)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
