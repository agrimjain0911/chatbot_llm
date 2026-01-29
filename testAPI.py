import requests
response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "message": "Hello kimi, what is my name",
        "session_id": "123233"
    }
)

print(response.status_code)
print(response.json())