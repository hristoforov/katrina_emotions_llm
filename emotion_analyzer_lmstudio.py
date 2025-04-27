import requests
import json
import time
import csv
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import os

# Debug flag to control console output
DEBUG = False
INFO = True

# Emotion labels
EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']

def analyze_emotion(text: str) -> str:
    """
    Analyze the emotion in the given text using LM Studio API.
    Returns one of: joy, sadness, anger, fear, surprise, love, neutral
    """
    try:
        systemPrompt = "You are an emotion classifier. Classify each text into one of: joy, sadness, anger, fear, love, surprise. Answer only with one word."
        userPrompt = f"Classify the following text:\n\"{text}\""

        # Prepare the request to LM Studio
        url = "http://192.168.100.132:1234/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.2-3b-instruct",
            "messages": [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.8
        }

        # Make the request
        if DEBUG:
            print("\nDebug - Sending request to LM Studio...")
            print(f"Debug - System prompt: {systemPrompt}")
            print(f"Debug - User prompt: {userPrompt}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # Parse the response with error handling
        result = response.json()
        if DEBUG:
            print(f"Debug - Raw response: {json.dumps(result, indent=2)}")
        
        # Validate response structure
        if not isinstance(result, dict) or 'choices' not in result or not result['choices']:
            if DEBUG:
                print("Debug - Invalid response structure: missing choices")
            return "unknown"
            
        if not isinstance(result['choices'], list) or len(result['choices']) == 0:
            if DEBUG:
                print("Debug - Invalid choices format or empty choices")
            return "unknown"
            
        if 'message' not in result['choices'][0]:
            if DEBUG:
                print("Debug - Missing message in choice")
            return "unknown"
            
        if 'content' not in result['choices'][0]['message']:
            if DEBUG:
                print("Debug - Missing content in message")
            return "unknown"

        predicted_token = result['choices'][0]['message']['content'].strip().lower()
        if DEBUG:
            print(f"Debug - Raw predicted token: '{predicted_token}'")

        # Clean up the prediction
        predicted_emotion = predicted_token.split()[0]  # Take only the first word
        predicted_emotion = predicted_emotion.strip('.,!?*:')  # Remove punctuation and special characters
        if DEBUG:
            print(f"Debug - Cleaned emotion: '{predicted_emotion}'")

        # If output is unexpected, default to unknown
        if predicted_emotion not in EMOTIONS:
            if DEBUG:
                print(f"Debug - Invalid emotion detected: '{predicted_emotion}'")
                print(f"Debug - Valid emotions: {EMOTIONS}")
            return "unknown"

        return predicted_emotion

    except Exception as e:
        if DEBUG:
            print(f"Debug - Error analyzing emotion: {str(e)}")
        return "unknown"

def load_data(file_path: str) -> List[Tuple[str, str]]:
    """Load data from a file where each line is 'text;emotion'"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, emotion = line.strip().split(';')
            data.append((text, emotion))
    return data

def evaluate_model(test_data: List[Tuple[str, str]]) -> Dict:
    """Evaluate the model on test data and return metrics"""
    total = len(test_data)
    correct = 0
    confusion_matrix = {e: {e2: 0 for e2 in EMOTIONS} for e in EMOTIONS}
    
    start_time = time.time()
    predictions = []
    true_labels = []

    for i, (text, true_emotion) in enumerate(test_data):
        if INFO:
            print(f"\nProcessing sample {i+1}/{total}")
            print(f"Text: {text}")
            print(f"True emotion: {true_emotion}")
        
        predicted = analyze_emotion(text)
        
        if INFO:
            print(f"Predicted emotion: {predicted}")
        
        if predicted == true_emotion:
            correct += 1
        
        # Only update confusion matrix if both emotions are valid
        if true_emotion in EMOTIONS and predicted in EMOTIONS:
            confusion_matrix[true_emotion][predicted] += 1
        elif DEBUG:
            print(f"Debug - Skipping confusion matrix update for invalid emotions: true={true_emotion}, predicted={predicted}")
        
        predictions.append(predicted)
        true_labels.append(true_emotion)

        # Add a small delay to avoid overwhelming the API
        #time.sleep(0.1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    accuracy = correct / total
    avg_time = total_time / total
    
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'total_time_seconds': total_time,
        'avg_time_per_sample': avg_time,
        'confusion_matrix': confusion_matrix,
        'true_labels' : true_labels,
        'predictions' : predictions
    }

def save_results(results: Dict, output_file: str):
    """Save evaluation results to a CSV file"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write basic metrics
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f"{results['accuracy']:.4f}"])
        writer.writerow(['Total Samples', results['total_samples']])
        writer.writerow(['Correct Predictions', results['correct_predictions']])
        writer.writerow(['Total Time (seconds)', f"{results['total_time_seconds']:.2f}"])
        writer.writerow(['Average Time per Sample (seconds)', f"{results['avg_time_per_sample']:.4f}"])
        
        # Write confusion matrix
        writer.writerow([])
        writer.writerow(['Confusion Matrix'])
        writer.writerow([''] + EMOTIONS)  # Header row
        
        for true_emotion in EMOTIONS:
            row = [true_emotion]
            for predicted_emotion in EMOTIONS:
                row.append(results['confusion_matrix'][true_emotion][predicted_emotion])
            writer.writerow(row)

def main():
    # Load test data
    test_data = load_data('data/test.txt')
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(test_data)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Total Time: {results['total_time_seconds']:.2f} seconds")
    print(f"Average Time per Sample: {results['avg_time_per_sample']:.4f} seconds")
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(results['true_labels'], results['predictions']))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(results['true_labels'], results['predictions']))
    
    # Save results
    output_file = 'emotion_analysis_results.csv'
    save_results(results, output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 