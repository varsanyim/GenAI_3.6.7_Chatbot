import pandas as pd
import requests
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


nltk.download('punkt_tab')

TOGETHER_API_KEY = "YOUR API KEY"  # Replace with your Together API key

# Custom Chatbot Project 
"""
Martina Wille

This project uses the Wikipedia API to source information for a custom chatbot. 
For this chatbot, we fetch an article of interest and use its content to enhance responses to user queries.

Basic question: What is the definition of an autocracy?

Other questions (example):
How does an autocracy differ from a democracy?
What are some historical examples of autocratic governments?
What are the key characteristics of an autocratic leader?
How does an autocracy maintain power?
What role does censorship play in an autocratic system?
How do autocracies impact human rights?
Can an autocratic government be beneficial?
What are some modern examples of autocracies?
How does an autocracy differ from a dictatorship?
What is the difference between an absolute monarchy and an autocracy?
How do autocracies handle opposition and dissent?
What are the economic consequences of an autocratic government?
How do autocratic leaders rise to power?
What role does propaganda play in autocratic regimes?
How have autocratic governments collapsed in the past?
What is the difference between a totalitarian state and an autocracy?
How does the international community respond to autocratic regimes?
Are there any legal frameworks that support autocracies?
What are some examples of transitions from autocracy to democracy?

"""

DEBUG = False

# Fetch Wikipedia Article
WIKI_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/Autocracy"
response = requests.get(WIKI_API_URL)
wikipedia_data = response.json()
article_text = wikipedia_data.get("extract", "No article content available.")

# Normalize text
article_text = article_text.lower()  # Convert to lowercase
article_text = re.sub(r"[^a-zA-Z0-9\s\.\,]", "", article_text)  # Remove special characters
# Tokenize sentences
sentences = sent_tokenize(article_text)

# Ensure at least 20 sentences
if len(sentences) < 20:
    sentences = sentences * (20 // len(sentences) + 1)
    sentences = sentences[:20]  

# Join back into a single cleaned text
processed_text = " ".join(sentences)

# Load Data into Pandas DataFrame
df = pd.DataFrame({"text": processed_text.split('. ')})  # Splitting into sentences
print("Dataset Loaded:")
for i, row in df.iterrows():
    print(f"{i}: {row['text']}")

# Together AI API Configuration
ENDPOINT = "https://api.together.xyz/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MAX_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1

def ask_together_ai(question, custom_prompt=None):
    prompt = custom_prompt if custom_prompt else question
    response = requests.post(
        ENDPOINT,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "stop": ["<|eot_id|>", "<|eom_id|>"],
            "stream": False,
        },
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    # Ensure response is valid
    if response.status_code != 200:
        return f"Error: API request failed with status {response.status_code}: {response.text}"

    response_json = response.json()  # Corrected: Now only calling .json() once
    if (DEBUG): print("API Response:", response_json) 

    # Extracting the content properly
    return response_json.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response from API.")

# Basic vs Custom Query Demonstration
basic_question = "What is the definition of an autocracy?"
custom_prompt = "Based on the following Wikipedia article on autocracy, provide a helpful response: \n" + "\n".join(df["text"]) + f"\nUser question: {basic_question}"


# Basic Answer
basic_answer = ask_together_ai(basic_question)
print("\n"*3+"="*80+"\nBasic Question: "+basic_question)
print("Basic Answer:", basic_answer)

# Custom Answer
custom_answer = ask_together_ai(basic_question, custom_prompt)
print("\n"*3+"="*80+"\nCustom Question: " f"Based on the following Wikipedia article on autocracy, provide a helpful response: {ENDPOINT} \n\tUser question: {basic_question}")
print("Custom Answer:", custom_answer)

# Interactive Chatbot Loop
print("\nType 'exit' to end chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    enhanced_prompt = f"Using the following Wikipedia article as a reference, answer the question: \n{df['text'].to_string(index=False)}\nUser: {user_input}"
    bot_response = ask_together_ai(user_input, enhanced_prompt)
    print("Bot:", bot_response)

