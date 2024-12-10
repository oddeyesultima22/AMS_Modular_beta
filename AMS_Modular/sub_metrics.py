import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import os

# Define the base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def tokenize_texts(tokenizer, texts, max_length=128):
    inputs = tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return inputs

def load_saved_model_and_tokenizer():
    try:
        # Define the model and tokenizer paths relative to the base directory
        model_path = os.path.join(BASE_DIR, 'sent_ana_model')
        tokenizer_path = os.path.join(BASE_DIR, 'sent_ana_model')

        # Load the model and tokenizer
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'TFBertForSequenceClassification': TFBertForSequenceClassification}
        )
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model or tokenizer: {e}")

def make_predictions(model, tokenizer, data, output_file):
    try:
        # Tokenize the input text data
        inputs = tokenize_texts(tokenizer, data['Text'])
        
        # Predict for all sub-metrics
        predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])
        sub_metrics = [
            'Thank Customer', 'Introduce Self', 'Ask Reason', 'Ask Accurate Details',
            'Ask Permission', 'Resolve Issue', 'Offer Assistance', 'Thank Again', 'Farewell'
        ]
        for idx, sub_metric in enumerate(sub_metrics):
            predicted_values = np.argmax(predictions[idx], axis=1)
            data[f'Predicted {sub_metric}'] = predicted_values
        
        # Save the predictions to a CSV file
        data.to_csv(output_file, index=False)
    except Exception as e:
        raise RuntimeError(f"Error making predictions: {e}")
