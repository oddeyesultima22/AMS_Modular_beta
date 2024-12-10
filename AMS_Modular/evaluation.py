import pandas as pd
import glob
import os

def convert_to_quality_category(avg_score):
    if avg_score >= 0.7:
        return 'Positive'
    elif 0.4 <= avg_score < 0.7:
        return 'Neutral'
    else:
        return 'Negative'

def calculate_scores(file_path):
    data = pd.read_csv(file_path)
    sub_metric_points = {
        'Thank Customer': 5, 'Introduce Self': 5, 'Ask Reason': 5,
        'Ask Accurate Details': 10, 'Ask Permission': 10, 'Resolve Issue': 50,
        'Offer Assistance': 5, 'Thank Again': 5, 'Farewell': 5
    }
    total_categories = 4
    unique_categories = data['Predicted Category'].nunique()
    category_percent = (unique_categories / total_categories) * 100

    quality_weights = {'Positive': 0.8, 'Neutral': 0.5, 'Negative': 0.2}
    weighted_quality = data['Predicted Quality'].map(quality_weights)
    average_weighted_quality = weighted_quality.mean()
    avg_quality_category = convert_to_quality_category(average_weighted_quality)
    average_predicted_sentiment = round(data['Predicted Sentiment'].mean(), 2)
    earned_points = {
        sub_metric: points if not data[data[f'Predicted {sub_metric}'] == 1].empty else 0
        for sub_metric, points in sub_metric_points.items()
    }
    overall_score = sum(earned_points.values())
    results = {
        'file': os.path.basename(file_path),
        'category %': round(category_percent, 2),
        'avg quality': avg_quality_category,
        'average_predicted_sentiment': average_predicted_sentiment,
        **earned_points,
        'Overall Score': overall_score
    }
    return results

def process_multiple_files(input_directory, output_file):
    file_paths = glob.glob(os.path.join(input_directory, '*.csv'))
    all_results = [calculate_scores(file) for file in file_paths]
    pd.DataFrame(all_results).to_csv(output_file, index=False)
