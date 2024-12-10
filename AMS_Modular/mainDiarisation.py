import os
from diarisation import diarize_audio
from transcription import transcribe_audio
from utils import get_text_with_timestamp, add_speaker_info_to_text, merge_sentence, save_to_csv

# Define the base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants with absolute paths
AUTH_TOKEN = "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"
INPUT_AUDIO_DIR = os.path.join(BASE_DIR, "input_audio")
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "data/raw")

def process_file(audio_file, output_csv):
    print(f"Processing file: {audio_file}")

    print("Starting transcription...")
    asr_result = transcribe_audio(audio_file)
    if not asr_result:
        print("Failed transcription.")
        return

    print("Starting diarization...")
    diarization_result = diarize_audio(audio_file, AUTH_TOKEN)
    if not diarization_result:
        print("Failed diarization.")
        return

    print("Merging transcription and diarization...")
    timestamp_texts = get_text_with_timestamp(asr_result)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    merged_text = merge_sentence(spk_text)

    print("Saving results to CSV...")
    save_to_csv(merged_text, output_csv)  # Use the utility function here

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

    # Process all audio files in the input directory
    for file in os.listdir(INPUT_AUDIO_DIR):
        if file.lower().endswith(('.wav', '.mp3')):
            input_path = os.path.join(INPUT_AUDIO_DIR, file)
            output_path = os.path.join(OUTPUT_CSV_DIR, f"{os.path.splitext(file)[0]}_text.csv")
            process_file(input_path, output_path)

if __name__ == "__main__":
    main()
