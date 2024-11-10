#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import time
import speech_recognition as sr
import whisper
import numpy as np

global stt

# Define a function to convert audio data to 16 kHz
def audio_data_to_array(audio_data):
    # Convert audio data to bytes, then to numpy array
    audio_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
    audio_array = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    return audio_array

# this is called from the background thread
def callback(recognizer, audio):
    print("Callback fired!")
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        #print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        #x = recognizer.recognize_whisper(audio,"base.en")
        # Convert audio data to 16kHz single-channel numpy array
        audio_array = audio_data_to_array(audio)

        result = stt.transcribe(audio_array)
        print("Result: ", result)
        print("Recognised: ", result["text"])
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))

r = sr.Recognizer()
m = sr.Microphone()
stt = whisper.load_model("base.en")
with m as source:
    print("adjust_for_ambient_noise")
    r.adjust_for_ambient_noise(source, duration=5)  # we only need to calibrate once, before we start listening
    r.dynamic_energy_threshold = True
print("Listening...")
# start listening in the background (note that we don't have to do this inside a `with` statement)
stop_listening = r.listen_in_background(m, callback)

# `stop_listening` is now a function that, when called, stops background listening

# do some unrelated computations for 5 seconds
for _ in range(500): time.sleep(0.1)  # we're still listening even though the main thread is doing other things

# calling this function requests that the background listener stop listening
stop_listening(wait_for_stop=False)
print("Exiting!")
# do some more unrelated things
#while True: time.sleep(0.1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping
