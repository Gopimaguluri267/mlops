import requests

test_data = {
    "data": [[45, "M", "NAP", 110, 0, 0, "Normal", 138, "N", -0.1, "Up"]]
}

response = requests.post(
    "http://localhost:8000/predict",
    json=test_data
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
