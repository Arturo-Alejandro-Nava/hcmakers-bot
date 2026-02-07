import requests

# --- CONFIG ---
# Paste your Render URL inside the quotes:
URL = "https://hcmakers-bot.onrender.com" 

print("--- CONNECTING TO HC MAKERS BOT ---")
print("(Type 'quit' to exit)")

while True:
    question = input("\nYou: ")
    if question.lower() == 'quit': break
    
    try:
        # We send the message to the /chat "room"
        response = requests.post(f"{URL}/chat", json={"message": question})
        
        if response.status_code == 200:
            print(f"Bot: {response.json().get('response')}")
        else:
            print("Bot is waking up... try again in 30 seconds.")
    except Exception as e:
        print(f"Connection Error: {e}")