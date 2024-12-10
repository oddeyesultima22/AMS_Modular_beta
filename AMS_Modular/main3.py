import os
import glob
import pandas as pd
from diarisation import diarize_audio
from transcription import transcribe_audio
from utils import get_text_with_timestamp, add_speaker_info_to_text, merge_sentence, save_to_csv
from predictions import load_fine_tuned_model, run_prediction_pipeline
from sub_metrics import load_saved_model_and_tokenizer, make_predictions
from evaluation import process_multiple_files

# Get the directory where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants
AUTH_TOKEN = "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"
INPUT_AUDIO_DIR = os.path.join(BASE_DIR, "input_audio")
RAW_CSV_DIR = os.path.join(BASE_DIR, "data/original")
PREDICTED_DIR = os.path.join(BASE_DIR, "data/predicted")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics")
EVALUATION_DIR = os.path.join(BASE_DIR, "data/evaluation")

def process_audio_files():
    """Process audio files: Transcription, Diarization, and Save CSV."""
    print("Starting Diarization and Transcription Pipeline...")
    for file in os.listdir(INPUT_AUDIO_DIR):
        if file.lower().endswith(('.wav', '.mp3')):
            input_path = os.path.join(INPUT_AUDIO_DIR, file)
            output_path = os.path.join(RAW_CSV_DIR, f"{os.path.splitext(file)[0]}_text.csv")
            print(f"Processing file: {file}")
            try:
                print('attempting asr result')
                asr_result = transcribe_audio(input_path)
                if not asr_result:
                    print("Failed transcription.")
                    continue
                print('attempting diarization result')
                diarization_result = diarize_audio(input_path, AUTH_TOKEN)
                if not diarization_result:
                    print("Failed diarization.")
                    continue
                print('attempting timestamp, spk_text and merge text')
                timestamp_texts = get_text_with_timestamp(asr_result)
                spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
                merged_text = merge_sentence(spk_text)
                save_to_csv(merged_text, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

def run_predictions_and_evaluations():
    """Run predictions and evaluations on the processed data."""
    print("Starting Prediction and Evaluation Pipeline...")
    os.makedirs(PREDICTED_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    # Load models
    model, tokenizer = load_fine_tuned_model()
    sub_model, sub_tokenizer = load_saved_model_and_tokenizer()

    # Predictions
    print("Running the predictions pipeline...")
    for file in glob.glob(f'{RAW_CSV_DIR}/*.csv'):
        data = pd.read_csv(file)
        output_file = os.path.join(PREDICTED_DIR, os.path.basename(file))
        run_prediction_pipeline(data, output_file, model, tokenizer)

    # Sub-metrics
    print("Making predictions...")
    for file in glob.glob(f'{PREDICTED_DIR}/*.csv'):
        data = pd.read_csv(file)
        output_file = os.path.join(METRICS_DIR, os.path.basename(file))
        make_predictions(sub_model, sub_tokenizer, data, output_file)

    # Evaluation
    print("Printing the evaluation outputs to csv...")
    process_multiple_files(METRICS_DIR, os.path.join(EVALUATION_DIR, 'combined_evaluation.csv'))

def main():
    os.makedirs(RAW_CSV_DIR, exist_ok=True)  # Ensure raw directory exists
    process_audio_files()                    # Step 1: Process audio files
    run_predictions_and_evaluations()        # Step 2: Run predictions and evaluations

if __name__ == "__main__":
    main()
