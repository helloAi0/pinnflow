import requests

# Test coordinates: directly behind the cylinder at t=5.0s
payload = {
    "points": [
        {"x": 1.0, "y": 0.0, "t": 5.0},
        {"x": 1.5, "y": 0.2, "t": 5.0},
        {"x": 2.0, "y": -0.2, "t": 5.0}
    ]
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)

if response.status_code == 200:
    print("Inference Success:")
    for i, pred in enumerate(response.json()["predictions"]):
        print(f"Point {i+1} -> U: {pred['u']:.4f}, V: {pred['v']:.4f}, P: {pred['p']:.4f}")
else:
    print(f"Error {response.status_code}: {response.text}")