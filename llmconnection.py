import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Did you set it in .env?")

# Initialize OpenAI client for LangChain
chat_model = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=api_key,
    temperature=0.7
)

# Dictionary to store memory per session
sessions_memory = {}


def clean_response(text: str) -> str:
    """
    Remove unwanted special characters from a text response.
    Keeps letters, numbers, basic punctuation, and spaces.
    Collapses multiple spaces into a single space.
    """
    # Keep letters, numbers, punctuation (. , ? ! : ; - ' "), and spaces
    cleaned = re.sub(r"[^a-zA-Z0-9.,?!:;'\-\s\"]+", "", text)

    # Replace multiple spaces or newlines with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove leading/trailing spaces
    cleaned = cleaned.strip()

    return cleaned


def process_message(message: str, session_id: str) -> str:
    """
    Send message to OpenAI LLM and return response.
    Each session_id keeps its own memory (conversation history).
    """
    try:
        # Initialize memory for this session if not exists
        if session_id not in sessions_memory:
            from langchain.memory import ConversationBufferMemory

            sessions_memory[session_id] = ConversationBufferMemory(
                memory_key="history", return_messages=True
            )

            # Preload context into memory
            context = """
# Context
1.) You are a mock interviewer.
2.) Act as a real interviewer.
3.) Test the candidate’s knowledge on the topic mentioned below.

# Task
1.) Conduct a mock interview for a candidate applying as a Java Spring Boot developer with 3 years of experience.
2.) The first three questions should be general:
   - Tell me about yourself
   - Describe the projects you’ve worked on
   - Explain your roles and responsibilities in that project
3.) After that, conduct a technical interview:
   - Core Java questions
   - Database-related questions
   - Testing-related questions
   - Real-world scenario and system design questions

# Constraints
1.) Be strict with answers; explanations should be clear.
2.) If the candidate answers something irrelevant, respond with:
   “Ok, we can go to the next question,” and then move on.
3.) Ask at least one scenario-based question and follow up on it.
4.) Start with basic questions, then move to intermediate and follow-up questions.
5.) Total questions should be between 25 and 30 (not more).
"""
            # Add the context as the first message in memory
            sessions_memory[session_id].chat_memory.add_user_message(context)
            sessions_memory[session_id].chat_memory.add_ai_message("Understood. Let's begin the interview.")

        memory = sessions_memory[session_id]

        # Create the conversation chain
        from langchain.chains import ConversationChain

        chain = ConversationChain(
            llm=chat_model,
            memory=memory,
            verbose=False
        )

        # Get the model’s response
        response = chain.run(message)

        # Clean unwanted characters
        response = clean_response(response)

        return response

    except Exception as e:
        return f"Error: {str(e)}"

