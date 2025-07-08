import requests

url = "http://localhost:8000/query"
headers = {"Content-Type": "application/json"}
data = {"question": "测试问题"}

response = requests.post(url, json=data, headers=headers)
print(response.json())