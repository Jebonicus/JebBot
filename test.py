import ollama

#client = ollama.Client()
#model = client.load_model('tinyllama')

stream = ollama.chat(model='tinyllama',
                       messages=[{ 'role': 'user', 'content': 'Why is the sky blue?' }],
                       stream=True)

for chunk in stream:
  #print(chunk['message']['content'], end='', flush=True)
  print(chunk['message']['content'], end='', flush=True)

#print(response['message']['content'])

#{ 'role': 'system', 'content': 'You are a very rude, sarcastic, evil AI assistant who gives quite concise answers. You keep you answers under 50 tokens.' },
                                 