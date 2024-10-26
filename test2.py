import ollama
import sounddevice as sd
import numpy as np
from piper.voice import PiperVoice
import whisper

def initSound(voiceModel):
    voicedir = "./" #Where onnx model files are stored on my machine
    model = voicedir+voiceModel
    voice = PiperVoice.load(model)
    text = "This is an example of text-to-speech using Piper TTS."

    # Setup a sounddevice OutputStream with appropriate parameters
    # The sample rate and channels should match the properties of the PCM data
    stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    stream.start()
    return voice, stream

def interactive_chat(model_name, system_prompt, voice, soundStream, stt):
    print("Welcome to the Ollama interactive chat! Type 'exit' to end the conversation.")
    
    messages=[ {'role': 'system', 'content': system_prompt}]
    while True:
        # Get user input
        try:
            user_input = input("You: ")

            # Check if the user wants to exit the chat
            if user_input.lower() == 'exit':
                print("Ending chat. Goodbye!")
                break
            user_json_object = {'role': 'user', 'content': user_input}
            messages.append(user_json_object)

            # Get response from the Ollama model
            try:
                stream = ollama.chat(model=model_name, messages=messages, stream=True)
                total_response=""
                first=True
                for chunk in stream:
                    chunk_response = chunk['message']['content']
                    total_response += chunk_response
                    print(("AI: " if first else "") + chunk_response, end='', flush=True)
                    first = False
                print("\n")
                

                for audio_bytes in voice.synthesize_stream_raw(total_response):
                    int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    soundStream.write(int_data)

                assistant_json_object = {'role': 'assistant', 'content': total_response}
                messages.append(assistant_json_object)

            except Exception as e:
                print("An error occurred:", e)
        except EOFError:
                print("Bye")

# Run the interactive chat with a specified model and system prompt
if __name__ == "__main__":
    # Modify the model_name and system_prompt as needed
    model_name = "tinyllama-rude"
    system_prompt = "You are a very rude, sarcastic, evil AI personality who gives quite concise answers."
    stt = whisper.load_model("base.en")

    voice,soundStream=initSound(voiceModel="en_GB-alan-medium.onnx")
    interactive_chat(model_name=model_name, system_prompt=system_prompt, voice=voice, soundStream=soundStream, stt=stt)
