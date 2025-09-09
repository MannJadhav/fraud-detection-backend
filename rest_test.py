# rest_test.py
import requests, json

url = "http://127.0.0.1:8000/predict"
payload = {
    "features": {
        "Amount": 100.0,
        "Time": 12345.0,
        # other features...
    },
    "card_id": "card_123"
}
resp = requests.post(url, json=payload)
print(resp.status_code, resp.text)
