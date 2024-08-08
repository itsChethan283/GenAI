import ollama

stream = ollama.chat(model="llama2", messages=[{"role": "user", "content":"Who is the president of india"}], stream=True)

for chunk in stream:
    print(chunk["message"]["content"], end="",  flush=True)