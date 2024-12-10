from pyannote.audio import Pipeline

def diarize_audio(audio_file, auth_token):
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=auth_token
        )
        diarization_result = pipeline(audio_file)
        return diarization_result
    except Exception as e:
        print(f"Error during diarization: {e}")
        return None
