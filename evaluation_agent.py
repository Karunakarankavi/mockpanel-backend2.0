import os
import json
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ---------------- Load Environment Variables ---------------- #
load_dotenv()

# Embedding model selection — ensure 1024-dim embeddings
EMBEDDING_MODEL = "text-embedding-3-small"  # 1024 dimensions

# ---------------- Initialize Pinecone ---------------- #
pc = Pinecone(api_key=os.getenv("PINECONE_API"))
INDEX_NAME = "topic-summary"

# Create index if it does not exist
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # 1024 dimensions for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)


# ---------------- EvaluationAgent ---------------- #
class EvaluationAgent:
    def __init__(self, role="Java Spring Boot Developer", experience_level="3 years"):
        self.role = role
        self.experience_level = experience_level
        self.topics = {}  # topic -> {score, summary, next_stage}
        self.current_topic = None
        self.questions_under_topic = []

    # ---------------- Add Q&A ---------------- #
    def add_question_answer(self, question: str, answer: str, topic: str, user_id: str):
        if not question.strip() or not answer.strip():
            print("⚠️ Skipping empty question or answer")
            return

        # Save Q&A embedding
        self._save_qna_embedding(user_id, topic, question, answer)

        # Check if topic changed
        if self.current_topic and topic.strip().lower() != self.current_topic.strip().lower():
            self._evaluate_topic(self.current_topic, self.questions_under_topic, user_id)
            self.questions_under_topic = []

        self.current_topic = topic
        self.questions_under_topic.append({"question": question, "answer": answer})

    # ---------------- Save Q&A Embedding ---------------- #
    def _save_qna_embedding(self, user_id: str, topic: str, question: str, answer: str):
        try:
            text = f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}"
            emb_response = openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
            vector = emb_response.data[0].embedding

            # Normalize embedding to 1024 dims (truncate or pad) if provider returns different size
            expected_dim = 1024
            if len(vector) != expected_dim:
                print(f"⚠️ Embedding length {len(vector)} != {expected_dim}. Normalizing (truncate/pad).")
                # Truncate if longer, pad with zeros if shorter
                if len(vector) > expected_dim:
                    vector = vector[:expected_dim]
                else:
                    vector = vector + [0.0] * (expected_dim - len(vector))

            vector_id = f"{user_id}-{topic}-{abs(hash(question))}"
            index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "user_id": user_id,
                        "topic": topic,
                        "question": question[:200],
                        "answer": answer[:200],
                    }
                }]
            )
            print(f"✅ Stored embedding for Q&A (topic='{topic}', id={vector_id})")
        except Exception as e:
            print(f"❌ Error saving Q&A to Pinecone: {e}")

    # ---------------- Evaluate Topic ---------------- #
    def _evaluate_topic(self, topic: str, qna_list: list, user_id: str):
        qna_text = "\n".join([f"Q: {q['question']}\nA: {q['answer']}" for q in qna_list])
        prompt = f"""
        You are a senior AI interviewer evaluating a candidate for the role of {self.role}
        with {self.experience_level} experience.

        You are currently assessing their understanding under the topic "{topic}".

        Here are all the questions and answers so far:
        {qna_text}

        Your task:
        1. Evaluate the candidate’s understanding level analytically.
        2. Determine if they are ready to move to deeper (twisted or advanced) questions.
        3. Suggest the appropriate next interview stage.

        Return a **strict JSON** with the following keys:
        - score (0–100): numerical score reflecting their grasp of this topic.
        - summary (3–5 lines): a professional summary describing overall performance, confidence, and clarity.
        - next_stage: one of ["basic", "intermediate", "advanced"], where:
            • "basic" → candidate needs simpler conceptual questions.
            • "intermediate" → candidate understood fundamentals; move to scenario-based or comparative questions.
            • "advanced" → candidate answered confidently; proceed to twisted or real-world design/application questions.
        - weak_areas: a concise list (array) of subtopics or concepts that need improvement.
        - next_focus: short guidance (1–2 lines) for what type of next question should be asked (e.g., "Ask about thread-safety and reflection in Singleton", "Move to Abstract Factory pattern", etc.)

        Example output:
        {{
          "score": 88,
          "summary": "The candidate demonstrated a strong understanding of Singleton and Factory patterns, including structure and use cases. They provided confident, well-structured answers.",
          "next_stage": "advanced",
          "weak_areas": ["Factory pattern Open/Closed Principle", "Thread-safety variations in Singleton"],
          "next_focus": "Ask scenario-based or real-world questions connecting Singleton and Factory patterns, such as their use in Spring Framework."
        }}

        Be analytical and precise. Avoid repeating the question text. Focus only on knowledge depth, accuracy, and readiness for the next level.
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert interviewer evaluating the candidate’s understanding."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            raw_output = response.choices[0].message.content.strip()
            try:
                feedback = json.loads(raw_output)
            except json.JSONDecodeError:
                feedback = {"score": 0, "summary": raw_output, "next_stage": "basic"}

            self.topics[topic] = feedback
            print(f"\n=== ✅ Topic Evaluation Completed: {topic} ===")
            print(f"Score: {feedback.get('score', 0)}")
            print(f"Next Stage: {feedback.get('next_stage', 'N/A')}")
            print(f"Summary: {feedback.get('summary', '')}\n")

            # Store topic summary embedding
            self._store_topic_summary(user_id, topic, feedback)

        except Exception as e:
            print(f"❌ Error evaluating topic {topic}: {e}")

    # ---------------- Store Topic Summary ---------------- #
    def _store_topic_summary(self, user_id: str, topic: str, feedback: dict):
        try:
            summary_text = (
                f"Topic: {topic}\n"
                f"Score: {feedback.get('score')}\n"
                f"Summary: {feedback.get('summary')}\n"
                f"Stage: {feedback.get('next_stage')}"
            )

            embedding_response = openai.embeddings.create(model=EMBEDDING_MODEL, input=summary_text)
            vector = embedding_response.data[0].embedding

            expected_dim = 1024
            if len(vector) != expected_dim:
                print(f"⚠️ Embedding length {len(vector)} != {expected_dim}. Normalizing (truncate/pad).")
                if len(vector) > expected_dim:
                    vector = vector[:expected_dim]
                else:
                    vector = vector + [0.0] * (expected_dim - len(vector))

            vector_id = f"{user_id}-{topic}-summary"
            index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "type": "summary",
                        "user_id": user_id,
                        "topic": topic,
                        "score": feedback.get("score"),
                        "summary": feedback.get("summary"),
                        "next_stage": feedback.get("next_stage"),
                    }
                }]
            )
            print(f"📊 Topic summary stored in Pinecone for '{topic}' (user={user_id})")

        except Exception as e:
            print(f"❌ Error storing topic summary: {e}")

    # ---------------- Finalize ---------------- #
    def finalize(self, user_id: str):
        if self.current_topic and self.questions_under_topic:
            self._evaluate_topic(self.current_topic, self.questions_under_topic, user_id)
        return self.topics
