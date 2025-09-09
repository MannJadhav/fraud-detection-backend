# test_client.py
import asyncio
import websockets
import json

async def test():
    uri = "ws://127.0.0.1:8000/ws/fraud"
    async with websockets.connect(uri) as ws:
        payload = {
            "features": {
                # use your column names from feature_columns.json
                "Amount": 100.0,
                "Time": 12345.0,
                # ... other features used by model (V1..V28 or others)
            },
            "card_id": "card_123"
        }
        await ws.send(json.dumps(payload))
        resp = await ws.recv()
        print("Response:", resp)

if __name__ == "__main__":
    asyncio.run(test())
