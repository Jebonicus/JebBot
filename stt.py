import whisper

model = whisper.load_model("base.en")
print("Loaded model")
result = model.transcribe("demo.wav")

print(result["text"])