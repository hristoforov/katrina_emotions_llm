import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import torch
import time

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
    
def setup_model():
    #model_name = "meta-llama/Llama-3.2-1B"  # Using LLaMA 3B
    #model_name = "lzw1008/Emollama-chat-7b"
    #model_name = "microsoft/phi-2"
    #model_name = "lzw1008/Emollama-chat-7b"
    model_name = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # ← Fix here

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
    
    return model, tokenizer



def evaluate_model(model, tokenizer, test_data):
    predictions = []
    true_labels = []
    results = []
    
    print("\nEvaluating model...\n")
    
    start_time = time.time()
    for _, row in test_data.iterrows():
        if DEBUG:
            print(f"Text: {row['text']}")
            print(f"True emotion: {row['emotion']}")
        
        predicted_emotion = analyze_emotion(row['text'], model, tokenizer)
        predictions.append(predicted_emotion)
        true_labels.append(row['emotion'])
        
        # Store results for CSV
        results.append({
            'text': row['text'],
            'true_emotion': row['emotion'],
            'predicted_emotion': predicted_emotion
        })
        
        if DEBUG:
            print(f"Predicted emotion: {predicted_emotion}\n")
    
    total_time = time.time() - start_time
    
    # Save results to CSV
    model_name = model.config._name_or_path.replace('/', '_')
    output_file = f"{model_name}.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
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

def analyze_emotion(text, model, tokenizer):
    try:
        # More structured prompt for Gemma
        prompt = (
f"""You are an emotion classifier. Your task is to classify the emotion in the following text into exactly one of these categories: joy, sadness, anger, fear, love, surprise.
Be accurate in your classification, choose the most appropriate emotion from the list.
Answer with exactly one word from the list above.
CLassify the following text:
{text}
"""
    
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate output with adjusted parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,            # Limit to just the emotion word
            temperature=0.2,             # Moderate randomness
            do_sample=True,              # Enable sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.9,                   # Standard top_p
            repetition_penalty=1.1        # Light repetition penalty
        )

        # Validate outputs
        if not isinstance(outputs, torch.Tensor) or outputs.numel() == 0:
            if DEBUG:
                print("Invalid output format")
            return "unknown"

        if len(outputs.shape) < 2:
            if DEBUG:
                print("Output tensor has incorrect shape")
            return "unknown"

        # Check if we got any output
        if len(outputs[0]) <= inputs["input_ids"].shape[-1]:
            if DEBUG:
                print("No new tokens generated")
            return "unknown"

        # Decode the generated token explicitly
        predicted_token = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        predicted_emotion = predicted_token.strip().lower()

        try:
            
            # Clean up the prediction
            predicted_emotion = predicted_emotion.split()[0]  # Take only the first word
            predicted_emotion = predicted_emotion.strip('.,!?')  # Remove punctuation

            # If output is unexpected, default to unknown
            valid_emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'love', 'neutral']
            if predicted_emotion not in valid_emotions:
                if DEBUG:
                    print(f"Invalid emotion detected: {predicted_emotion}")
                return "unknown"

            return predicted_emotion
        except Exception as decode_error:
            if DEBUG:
                print(f"Error decoding output {predicted_emotion}: {str(decode_error)} ")
            return "unknown"

    except Exception as e:
        if DEBUG:
            print(f"Error analyzing emotion: {str(e)}")
        return "unknown"
    
def main():
    # Load data
    print("Loading data...")
    df =  load_data('data/test.txt')
    
    # Set up model
    print("Setting up model...")
    model, tokenizer = setup_model()
    
    # Use more samples for testing (50 samples)
    #test_data = df.head(50)  # Use first 50 samples after shuffling
    test_data = df
    evaluate_model(model, tokenizer, test_data)

if __name__ == "__main__":
    main()