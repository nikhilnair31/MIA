import os
import requests
from dotenv import main

main.load_dotenv()
client_id = str(os.getenv("ACCESS_TOKEN"))
client_secret = str(os.getenv("ACCESS_SECRET"))

url = 'https://oauth.fatsecret.com/connect/token'

headers = {'content-type': 'application/x-www-form-urlencoded'}

data = {
    'grant_type': 'client_credentials',
    'scope': 'basic'
}

response = requests.post(
    url,
    headers=headers,
    data=data,
    auth=(client_id, client_secret)
)

print(response.json())

resp_dict = response.json()
access_token = resp_dict["access_token"]
print(access_token)

params = {
    'method': 'food_entries.get_monthly_most_eaten',
    'format': 'json',
    'oauth_token': access_token
}

response = requests.get(url, params=params)

if response.ok:
    data = response.json()
    print(data)
else:
    print('error')