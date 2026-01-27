import speech_recognition as sr
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def main():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2

        SYSTEM_PROMPT = """
            You are an expert voice agent and you generate the text based on 
            the transcript of what user has said 
            you need to output as you are a voice agent and whatever you speak will be 
            converted back to audio.
"""
        message_history = [{"role":"system","content":SYSTEM_PROMPT}]
        while True:

            print("Speak something...")
            audio = r.listen(source)

            print("Processing audio......")
            stt = r.recognize_google(audio)

            print("You said:",stt)

            message_history.append({"role":"user","content":stt})
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=message_history
            )

            result = response.choices[0].message.content
            print(f"AI:{result}")

            message_history.append({"role":"assistant","content":result})

main()