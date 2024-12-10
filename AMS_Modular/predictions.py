import pandas as pd
import numpy as np
import tensorflow as tf
import transformers
import os
from transformers import TFBertModel, BertTokenizer

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def scale_sentiment_output(x):
    return 0.1 + 0.8 * x

def load_fine_tuned_model():
    try:
        # Define model and tokenizer paths relative to the base directory
        model_path = os.path.join(BASE_DIR, 'fine_tuned_bert', 'saved_model')
        tokenizer_path = os.path.join(BASE_DIR, 'fine_tuned_bert')

        # Load the fine-tuned model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TFBertModel': TFBertModel,
                'scale_sentiment_output': scale_sentiment_output
            }
        )
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model or tokenizer: {e}")

def tokenize_texts(tokenizer, texts, max_length=128):
    inputs = tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        return_tensors="tf",
        max_length=max_length
    )
    return inputs

def run_prediction_pipeline(data, output_file, model, tokenizer):
    # Tokenize texts
    inputs = tokenize_texts(tokenizer, data['Text'])
    
    # Make predictions
    predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])
    predicted_categories = np.argmax(predictions['category_output'], axis=1)
    predicted_qualities = np.argmax(predictions['quality_output'], axis=1)
    predicted_sentiments = predictions['sentiment_output'].flatten()

    # Define mappings for predictions
    inverse_category_mapping = {
        0: 'Greetings',
        1: 'Problem Investigation',
        2: 'Closure',
        3: 'Account Verification'
    }
    inverse_quality_mapping = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}

    # Add predictions to the dataframe
    data['Predicted Category'] = pd.Series(predicted_categories).map(inverse_category_mapping)
    data['Predicted Quality'] = pd.Series(predicted_qualities).map(inverse_quality_mapping)
    data['Predicted Sentiment'] = predicted_sentiments.round(2)

    # Save predictions to CSV
    data = data[['Person', 'Text', 'Predicted Category', 'Predicted Quality', 'Predicted Sentiment']]
    data.to_csv(output_file, index=False)
