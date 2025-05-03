import requests

headers = {
    "Authorization": "Bearer "
}
response = requests.get(
    "https://huggingface.co/api/whoami-v2", headers=headers, verify=False)
print(response.json())
