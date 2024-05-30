import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gradio as gr

def speech_to_text(audio):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    try:
        # Convert audio to WAV format
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
        
        # Recognize speech
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand audio."
    except sr.RequestError as e:
        return "Could not request results; {0}".format(e)

def analyze_sentiment(text):
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    scores = sid.polarity_scores(text)
    return scores

def main(audio):
    # Get speech input
    text = speech_to_text(audio)
    
    # Perform sentiment analysis
    sentiment_scores = analyze_sentiment(text)
    max_score = max(sentiment_scores.values())
    
    sentiment = ""
    for key, val in sentiment_scores.items():
        if val == max_score:
            if key == "neg":
                sentiment = "Negative"
            elif key == "neu":
                sentiment = "Neutral"
            elif key == "pos":
                sentiment = "Positive"
            elif key == "compound":
                sentiment = "Compound"
            else:
                sentiment = "Unknown"
            break
    
    return f"You said: {text}\nSentiment: {sentiment}"

# Create the Gradio interface
iface = gr.Interface(
    fn=main,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Speech-to-Text and Sentiment Analysis",
    description="Upload an audio file or record using the microphone. The recognized text and sentiment analysis will be displayed below.",
)

if __name__ == "__main__":
    iface.launch(share=True)