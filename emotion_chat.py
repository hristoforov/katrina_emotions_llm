import requests
import json

# Debug flag to control console output
DEBUG = False

# Emotion to emoji mapping
EMOTION_EMOJI = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😮',
    'love': '❤️',
    'neutral': '😐',
    'unknown': '❓'
}

def analyze_emotion(text):
    try:
        systemPrompt = "You are an emotion classifier. Classify each text into one of: joy, sadness, anger, fear, love, surprise. Answer only with one word."
        # Prepare the prompt
        userPrompt = f"Classify the following text:\n\"{text}\""

        # Prepare the request to LM Studio
        url = "http://192.168.100.132:1234/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.2-3b-instruct",  # optional, but good to be explicit
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
        valid_emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'love', 'neutral']
        if predicted_emotion not in valid_emotions:
            if DEBUG:
                print(f"Debug - Invalid emotion detected: '{predicted_emotion}'")
                print(f"Debug - Valid emotions: {valid_emotions}")
            return "unknown"

        return predicted_emotion

    except Exception as e:
        if DEBUG:
            print(f"Debug - Error analyzing emotion: {str(e)}")
        return "unknown"

def main():
    print("Emotion Chat - Type 'quit' to exit")
    print("I'll analyze your text and respond with an emoji representing the emotion.")
    print("Available emotions: joy, sadness, anger, fear, surprise, love, neutral")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for quit command
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
            
        # Skip empty input
        if not user_input:
            continue
            
        # Analyze emotion
        emotion = analyze_emotion(user_input)
        
        # Get corresponding emoji
        emoji = EMOTION_EMOJI.get(emotion, EMOTION_EMOJI['unknown'])
        
        # Print response
        print(f"Bot: {emoji} ({emotion})")

if __name__ == "__main__":
    main() 