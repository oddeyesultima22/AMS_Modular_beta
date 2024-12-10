import os
import glob
import pandas as pd
from predictions import load_fine_tuned_model, run_prediction_pipeline
from sub_metrics import load_saved_model_and_tokenizer, make_predictions
from evaluation import process_multiple_files

# Define the base directory relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directory paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
ORIGINAL_DIR = os.path.join(DATA_DIR, 'original')
PREDICTED_DIR = os.path.join(DATA_DIR, 'predicted')
METRICS_DIR = os.path.join(DATA_DIR, 'metrics')
EVALUATION_DIR = os.path.join(DATA_DIR, 'evaluation')

def main():
    # Ensure directories exist
    os.makedirs(PREDICTED_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    # Load fine-tuned model and tokenizer
    model, tokenizer = load_fine_tuned_model()

    # Run predictions on original CSV files
    for file in glob.glob(os.path.join(ORIGINAL_DIR, '*.csv')):
        data = pd.read_csv(file)
        output_file = os.path.join(PREDICTED_DIR, os.path.basename(file))
        run_prediction_pipeline(data, output_file, model, tokenizer)

    # Load sub-metrics model and tokenizer
    sub_model, sub_tokenizer = load_saved_model_and_tokenizer()

    # Run sub-metrics predictions
    for file in glob.glob(os.path.join(PREDICTED_DIR, '*.csv')):
        data = pd.read_csv(file)
        output_file = os.path.join(METRICS_DIR, os.path.basename(file))
        make_predictions(sub_model, sub_tokenizer, data, output_file)

    # Process evaluation files
    process_multiple_files(METRICS_DIR, os.path.join(EVALUATION_DIR, 'combined_evaluation.csv'))

if __name__ == "__main__":
    main()
