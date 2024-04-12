"""Function for recording audio from a microphone."""
import io
import typing
import time
import wave
from pathlib import Path
from typing import Union
import scipy
from transformers import AutoProcessor, BarkModel
from rhasspysilence import WebRtcVadRecorder, VoiceCommand, VoiceCommandResult
import pyaudio
from os import PathLike
from deepgram import Deepgram
import asyncio
import whisper
import os
import re
from transformers import pipeline
from transformers import AutoProcessor, AutoModel


pa = pyaudio.PyAudio()

class Audio():
    def __init__(self, model: str = "facebook/wav2vec2-base-960h", deepgram_key: str = None) -> None:
        # self.deepgram = Deepgram(deepgram_key)
        pass

    def buffer_to_wav(self, buffer: bytes) -> bytes:
        """Wraps a buffer of raw audio data in a WAV"""
        rate = int(16000)
        width = int(2)
        channels = int(1)

        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
            with wav_file:
                wav_file.setframerate(rate)
                wav_file.setsampwidth(width)
                wav_file.setnchannels(channels)
                wav_file.writeframesraw(buffer)

            return wav_buffer.getvalue()
        
    def transcribe(self, file_name):
        model = whisper.load_model("base")
        result = model.transcribe(file_name)
        
        return re.sub(r'[^\w\s]', "", result["text"])

    def speech_to_text(self, max_seconds = None, recording_path = None) -> None:
        """
        Records audio until silence is detected
        Saves audio to audio/recording.wav
        """
        recorder = WebRtcVadRecorder(
            vad_mode=3,
            silence_seconds=6,
            max_seconds=max_seconds
        )
        recorder.start()
        # file directory
        wav_sink = "audio/audio_files/"
        # file name
        wav_filename = "recording"
        if wav_sink:
            wav_sink_path = Path(wav_sink)
            if wav_sink_path.is_dir():
                # Directory to write WAV files
                wav_dir = wav_sink_path
            else:
                # Single WAV file to write
                wav_sink = open(wav_sink, "wb")
        voice_command: typing.Optional[VoiceCommand] = None
        audio_source = pa.open(
            rate=16000,
            format=pyaudio.paInt16,
            channels=1,
            input=True,
            frames_per_buffer=960,
        )
        audio_source.start_stream()

        try:
            chunk = audio_source.read(960)
            while chunk:
                # Look for speech/silence
                voice_command = recorder.process_chunk(chunk)

                if voice_command:
                    _ = voice_command.result == VoiceCommandResult.FAILURE
                    # Reset
                    audio_data = recorder.stop()
                    if wav_dir:
                        # Write WAV to directory
                        wav_path = (wav_dir / time.strftime(wav_filename)).with_suffix(
                            ".wav"
                        )
                        wav_bytes = self.buffer_to_wav(audio_data)
                        wav_path.write_bytes(wav_bytes)
                        break
                    elif wav_sink:
                        # Write to WAV file
                        wav_bytes = self.buffer_to_wav(audio_data)
                        wav_sink.write(wav_bytes)
                # Next audio chunk
                chunk = audio_source.read(960)

        finally:
            try:
                audio_source.close_stream()
            except Exception:
                pass

            string_words = self.transcribe(recording_path)
            
        return string_words

        
    def text_to_speech(self, sentence):
                
        processor = AutoProcessor.from_pretrained("suno/bark-small")
        model = AutoModel.from_pretrained("suno/bark-small")

        inputs = processor(
            text=["Hello"],
            return_tensors="pt",
        )

        speech_values = model.generate(**inputs, do_sample=True)

        sampling_rate = model.config.sample_rate
        scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())




if __name__ == "__main__":
    audio = Audio()
    # audio.speech_to_text()
    audio.text_to_speech("hello world")
