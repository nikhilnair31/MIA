import os
import openai
from dotenv import main
from fatsecret import Fatsecret

main.load_dotenv()
gpt3_api_key = str(os.getenv("OPENAI_API_KEY"))
consumer_key = str(os.getenv("CONSUMER_KEY"))
consumer_secret = str(os.getenv("CONSUMER_SECRET"))
access_token = str(os.getenv("ACCESS_TOKEN"))
access_secret = str(os.getenv("ACCESS_SECRET"))

fs = Fatsecret(consumer_key, consumer_secret)
openai.api_key = gpt3_api_key

search_text = 'Per 100g'
foodname_prompt = 'Format the provided text as shown "ABC : Xg".\n'
transcript_text = 'alright make a food log entry. white rice 260 grams dal 90 grams fried mackarel 80 grams.'
print(f'Transcript: {transcript_text}\n')

# def log_entry_foods_serach():
#     if 'log' in transcript_text and 'entry' in transcript_text:
#         print(f'User asked to make log entry\n')

#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "system", "content": foodname_prompt+transcript_text}]
#         )
#         response_text = response["choices"][0]["message"]["content"]
#         print(f'Response:\n{response_text}\n')

#         lines = [line.strip() for line in response_text.split('\n') if line.strip()]
#         food_items = []
#         for line in lines:
#             parts = line.split(':')
#             food_item = {'name': parts[0].strip(), 'amount': parts[1].strip()}
#             food_items.append(food_item)

#         print(f'food_items:\n{food_items}\n')

#         for food in food_items:
#             foods = fs.foods_search(food["name"])
#             print(f'foods: {foods}\n')

#             for i, f in enumerate(foods):
#                 if search_text in f["food_description"]:
#                     print(f"Found '{search_text}' in dictionary at index {i}")
#                     break
            
#             print(f'final food: {foods[i]}\n')

fs.get_authorize_url()
fs.authenticate()

# foods = client.foods_most_eaten(20)
# for i, food in enumerate(foods):
#     print(f"{i+1}. {food['food_name']}: {food['times_eaten']} times")