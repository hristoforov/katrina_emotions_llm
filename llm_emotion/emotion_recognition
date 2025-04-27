import pandas as pd
import requests
import json
import time
from sklearn.metrics import classification_report, confusion_matrix

# Debug flag to control console output
DEBUG = True

def load_data(file_path):
    """Load and preprocess the emotion dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, emotion = line.strip().split(';')
            data.append({'text': text, 'emotion': emotion})
    return pd.DataFrame(data)

def analyze_emotion(text):
    try:
        # Prepare the prompt
        prompt = f"""You are an emotion classifier. Your task is to classify the emotion in the following text into exactly one of these categories: joy, sadness, anger, fear, love, surprise.
Be accurate in your classification, choose the most appropriate emotion from the list.
Classify the following text:\n
\"{text}\"\n
Answer with exactly one word from the list above.

"""

        # Prepare the request to LM Studio
        url = "http://192.168.100.132:1234/v1/completions"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt,
            "max_tokens": 5,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["\n\n"]
        }

        # Make the request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        result = response.json()
        predicted_token = result['choices'][0]['text'].strip().lower()

        # Clean up the prediction
        predicted_emotion = predicted_token.split()[0]  # Take only the first word
        predicted_emotion = predicted_emotion.strip('.,!?')  # Remove punctuation

        # If output is unexpected, default to unknown
        valid_emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'love', 'neutral']
        if predicted_emotion not in valid_emotions:
            if DEBUG:
                print(f"Invalid emotion detected: {predicted_token}")
            return "unknown"

        return predicted_emotion

    except Exception as e:
        if DEBUG:
            print(f"Error analyzing emotion: {str(e)}")
        return "unknown"

def evaluate_model(test_data):
    predictions = []
    true_labels = []
    
    print("\nEvaluating model...\n")
    
    start_time = time.time()
    for _, row in test_data.iterrows():
        if DEBUG:
            print(f"Text: {row['text']}")
            print(f"True emotion: {row['emotion']}")
        
        predicted_emotion = analyze_emotion(row['text'])
        predictions.append(predicted_emotion)
        true_labels.append(row['emotion'])
        
        if DEBUG:
            print(f"Predicted emotion: {predicted_emotion}\n")
    
    total_time = time.time() - start_time
    
    # Save results to CSV
    results = [{'text': row['text'], 'true_emotion': row['emotion'], 'predicted_emotion': pred} 
              for row, pred in zip(test_data.to_dict('records'), predictions)]
    pd.DataFrame(results).to_csv('lmstudio_results.csv', index=False)
    print("\nResults saved to lmstudio_results.csv")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    # Print timing statistics
    print("\nTiming Statistics:")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Number of samples: {len(test_data)}")
    print(f"Average time per sample: {total_time/len(test_data):.2f} seconds")

def main():
    # Load data
    print("Loading data...")
    df = load_data('data/test.txt')
    
    # Use all samples for testing
    test_data = df
    evaluate_model(test_data)

if __name__ == "__main__":
    main() 