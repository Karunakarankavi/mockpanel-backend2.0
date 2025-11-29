from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from texttospeech import tts

# Connect to Ollama
llm = OllamaLLM(model="llama2")

# Create chat prompt with memory slot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional interview panel member. "
 "Answer concisely, factually, and without roleplay, emojis, or stage directions. "
 "Do not include actions like *smiles*, *adjusts glasses*, etc. "
 "Only respond with plain text answers"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# Combine LLM + prompt
chain = prompt | llm

# Store memory (per session)
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Runnable with memory
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Session config
config = {"configurable": {"session_id": "karunakaran"}}

print("💬 Chat with your local LLaMA2 (type 'exit' to quit)\n")

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    response = conversation.invoke({"input": user_input}, config)
    tts(response)
