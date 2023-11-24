import openai
import time

openai.api_base = "http://localhost:1234/v1"  # point to the local server
openai.api_key = ""  # no need for an API key

system_prompt = """
Your name is MIA and you're an AI companion of the user. Keep your responses short. This is your first boot up and your first interaction with the user so ensure that you ask details about them to remember for the future. This includes things like their name, job/university, residence etc. Ask anything about them until you think it's enough or they stop you.
Internally you have the personality of JARVIS and Chandler Bing combined. You tend to make sarcastic jokes and observations. Do not patronize the user but adapt to how they behave with you.
You help the user with all their requests, questions and tasks. Be honest and admit if you don't know something when asked. 
"""
message_context = []
message_context.append({
    "role": "system", "content": system_prompt
})

def llm_call(input_msg):
    completion = openai.ChatCompletion.create(
        model="local-model",  # this field is currently unused
        messages=message_context,
        temperature=1,
        max_tokens=512,
    )
    return completion

def add_context(source, msg):
    global message_context
    message_context.append({
        "role": source, "content": msg
    })

def chat_with_bot():
    while True:
        inp = input("Input:\n")

        if inp.lower() == "/bye":
            print("Goodbye!")
            break

        start_time = time.time()

        add_context('user', input_msg)
        completion = llm_call(inp)

        response = completion.choices[0].message.content
        add_context('assistant', response)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'Response:\n{response}\nGen Time: {elapsed_time:.2f}s\n')

if __name__ == "__main__":
    chat_with_bot()
