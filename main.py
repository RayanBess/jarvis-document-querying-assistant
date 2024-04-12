"""Main file for the Jarvis project"""
import os
from time import time
import asyncio
from RAG import query_from_disk
from dotenv import load_dotenv
import openai
import pygame
from pygame import mixer
import elevenlabs
from elevenlabs import Voice, VoiceSettings
import speech_recognition as sr
from transformers import pipeline
from audio import Audio
from image_rec import VLM


# Load API keys
load_dotenv()

# Instantiate the Audio class
au = Audio(deepgram_key= os.getenv("DEEPGRAM_API_KEY"))
vlmd = VLM()
RECORDING_PATH = "audio/audio_files/recording.wav"
elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))


# mixer is a pygame module for playing audio
mixer.init()



def log(log: str):
    """
    Print and write to status.txt
    """
    print(log)
    with open("status.txt", "w") as f:
        f.write(log)


if __name__ == "__main__":
    string_words = ''
    
    while "hey jarvis" not in string_words.lower():

    # Record audio
        log("Listening...")
        string_words = au.speech_to_text(max_seconds=5, recording_path=RECORDING_PATH)
        log("Done listening")

        
        current_time = time()

    sound = mixer.Sound("audio/audio_files/welcome_message.wav")
    sound.play()
    
    while "jarvis turn off" not in string_words.lower():
        # Record audio
        log("Listening...")
        string_words = au.speech_to_text(recording_path=RECORDING_PATH)
        log("Done listening")

        # Transcribe audio
        
        if "jarvis turn off" in string_words.lower():
            continue
        
        with open("conv.txt", "a") as f:
            f.write(f"{string_words}\n")
        transcription_time = time() - current_time
        log(f"Finished transcribing in {transcription_time:.2f} seconds.")

        # Get response from query
        current_time = time()
        response2 = query_from_disk(string_words)
        print(response2)
        # context += response
    #     gpt_time = time() - current_time
    #     log(f"Finished generating response in {gpt_time:.2f} seconds.")

    #     # Convert response to audio
    #     current_time = time()
    #     audio = elevenlabs.generate(
    #         text=str(response2), voice=Voice(
    #     voice_id='UrE8mK37ssJ5yYNuQGZM',
    #     settings=VoiceSettings(stability=0.71, similarity_boost=0.9, style=0.0, use_speaker_boost=True)
    # ), model="eleven_monolingual_v1"
    #     )
    #     elevenlabs.save(audio, "audio/audio_files/response.wav")
    #     audio_time = time() - current_time
    #     log(f"Finished generating audio in {audio_time:.2f} seconds.")

    #     # Play response
    #     log("Speaking...")
    #     sound = mixer.Sound("audio/audio_files/response.wav")
    #     # Add response as a new line to conv.txt
    #     with open("conv.txt", "a") as f:
    #         f.write(f"{response2}\n")
    #     sound.play()
    #     pygame.time.wait(int(sound.get_length() * 1000))
    #     print(f"\n --- USER: {string_words}\n --- JARVIS: {response2}\n")
