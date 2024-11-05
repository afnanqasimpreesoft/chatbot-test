import requests

url = "http://127.0.0.1:8000/generate"
prompt = {"prompt": "Can you tell me about Python?"}

response = requests.post(url, json=prompt)
print(response.json())
