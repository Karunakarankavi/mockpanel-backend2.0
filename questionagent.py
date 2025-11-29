import os
import json
import openai
import redis
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from threading import Lock

from pinecone import Pinecone

from evaluation_agent import EvaluationAgent

# ---------------- Redis Setup ---------------- #
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# ---------------- Flask Setup ---------------- #
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API"))  # Your Pinecone key in .env
INDEX_NAME = "topic-summary"  # must match your Pinecone index name
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
agent_lock = Lock()
agents = {}
evaluators = {}  # user_id -> EvaluationAgent instance


# ---------------- Core Class ---------------- #
class QuestionPatternAgent:
    evaluators = {}  # user_id -> EvaluationAgent instance

    def __init__(self, question_structure, developer_role, experience_level, max_questions_per_topic=2, user_id=None):
        self.structure = question_structure
        self.developer_role = developer_role
        self.experience_level = experience_level
        self.max_questions_per_topic = max_questions_per_topic
        self.user_id = user_id

        self.current_domain = list(self.structure.keys())[0]
        self.current_topic_index = 0
        self.current_pattern_index = 0
        self.question_count = 0
        self.topics = list(self.structure[self.current_domain].keys())

    def _get_current_topic(self):
        return self.topics[self.current_topic_index]

    def _get_current_pattern(self):
        domain, topic = self.current_domain, self._get_current_topic()
        patterns = self.structure[domain][topic]
        return patterns[self.current_pattern_index % len(patterns)]

    def _move_to_next_topic(self):
        self.current_topic_index += 1
        self.current_pattern_index = 0
        self.question_count = 0

        if self.current_topic_index >= len(self.topics):
            domain_keys = list(self.structure.keys())
            current_domain_index = domain_keys.index(self.current_domain)
            if current_domain_index + 1 < len(domain_keys):
                self.current_domain = domain_keys[current_domain_index + 1]
                self.topics = list(self.structure[self.current_domain].keys())
                self.current_topic_index = 0
            else:
                self.current_domain = None

    def _get_asked_questions(self, topic):
        """Fetch already asked questions for this user & topic."""
        redis_key = f"asked_questions:{self.user_id}:{topic}"
        data = redis_client.lrange(redis_key, 0, -1)
        return data if data else []

    def _store_asked_question(self, topic, question):
        """Store the new question in Redis for memory."""
        redis_key = f"asked_questions:{self.user_id}:{topic}"
        redis_client.rpush(redis_key, question)

    def _embed_text(self, text):
        """Generate embeddings for text using OpenAI's embedding model and normalize to 1024 dimensions."""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            vector = response.data[0].embedding

            # ---------------- Ensure correct dimension (1024) ---------------- #
            expected_dim = 1024
            if len(vector) != expected_dim:
                print(f"⚠️ Embedding length {len(vector)} != {expected_dim}. Normalizing (truncate/pad).")
                if len(vector) > expected_dim:
                    vector = vector[:expected_dim]
                else:
                    vector = vector + [0.0] * (expected_dim - len(vector))

            return vector

        except Exception as e:
            print(f"⚠️ Embedding generation error: {e}")
            return None

    def _generate_question_from_llm(self, domain, topic, pattern_type, previous_answer=None):
        # ---------------- Retrieve topic summary & weak areas from Pinecone ---------------- #
        try:
            index = pc.Index("topic-summary")  # Pinecone index name

            # Get embedding for the topic to query relevant summary
            topic_vector = self._embed_text(topic)

            results = index.query(
                vector=topic_vector,
                top_k=1,
                include_metadata=True,
                filter={"type": "summary"}  # Filter ensures we only get summary type data
            )
            print(results)
            topic_summary = ""
            weak_areas = []

            if results.matches:
                metadata = results.matches[0].metadata
                topic_summary = metadata.get("summary", "")
                weak_areas = metadata.get("weak_areas", [])
            else:
                print(f"No summary found in Pinecone for topic: {topic}")

        except Exception as e:
            print(f"⚠️ Pinecone retrieval error: {e}")
            topic_summary = ""
            weak_areas = []

        # ---------------- Build context based on retrieved data ---------------- #
        summary_context = ""
        if topic_summary or weak_areas:
            weakness_text = ", ".join(weak_areas) if weak_areas else "N/A"
            summary_context = f"""
    Candidate's previous performance summary on this topic:
    "{topic_summary}"

    Focus areas for improvement: {weakness_text}
    """

        # ---------------- Dynamic follow-up based on previous answer ---------------- #
        dynamic_hint = ""
        if previous_answer:
            dynamic_hint = f"""
    The candidate previously answered: "{previous_answer}".
    Generate a follow-up or related question to explore deeper understanding.
    """
        # ---------------- Construct final LLM prompt ---------------- #
        prompt = f"""
    You are a professional interviewer for the role of {self.developer_role}.
    Generate one **{pattern_type} interview question** for a candidate with {self.experience_level} experience.
    Topic: "{topic}" under {domain}.
    {summary_context}
    {dynamic_hint}
    Focus the question specifically on weak or unclear areas to help assess improvement.
    Return only the question — no explanations or answers.
    """

        # ---------------- Call OpenAI model ---------------- #
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict technical interviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            question = response.choices[0].message.content.strip()
            return question

        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"

    def get_question(self, previous_answer=None):
        """Main question generation with duplicate prevention."""
        if not self.current_domain:
            return {"question": "✅ All topics completed."}

        domain, topic = self.current_domain, self._get_current_topic()
        pattern_type = self._get_current_pattern()
        use_previous_answer = previous_answer if self.question_count > 0 else None

        asked_questions = self._get_asked_questions(topic)
        attempt = 0

        while attempt < 3:
            question = self._generate_question_from_llm(domain, topic, pattern_type, use_previous_answer)
            if question not in asked_questions:
                break
            attempt += 1

        # store new question in Redis
        self._store_asked_question(topic, question)

        self.question_count += 1
        self.current_pattern_index += 1

        if self.question_count >= self.max_questions_per_topic:
            self._move_to_next_topic()

        return {"domain": domain, "topic": topic, "pattern": pattern_type, "question": question}


# ---------------- Endpoint ---------------- #
# question_structure = {
#     "Java": {
#         "Basic Design Patterns (Singleton, Factory)": ["Definition-based", "Scenario-based", "Real-world usage-based"],
#         "Basic Input/Output (I/O) operations": ["Definition-based", "Scenario-based", "Real-world usage-based"]
#     },
#     "Spring Boot": {
#         "Dependency Injection": ["Definition-based", "Scenario-based", "Real-world usage-based"],
#         "Creating RESTful APIs": ["Definition-based", "Scenario-based", "Real-world usage-based"]
#     }
# }

question_asked = None
def get_question_endpoint(user_answer, userid):
    global question_asked
    user_id = userid
    previous_answer = user_answer

    # ---------------- Ensure QuestionPatternAgent ---------------- #
    data = redis_client.get(user_id)
    payload = json.loads(data)
    question_structure = payload.get("question")
    role = payload.get("role")
    exp = payload.get("experience")


    with agent_lock:
        if user_id not in agents:
            agents[user_id] = QuestionPatternAgent(
                question_structure,
                developer_role=role,
                experience_level=exp,
                user_id=user_id
            )

    agent = agents[user_id]
    result = agent.get_question(previous_answer)

    # ---------------- Ensure EvaluationAgent ---------------- #
    if user_id not in evaluators:
        print("not present")
        print("*********************user id ************" , user_id )
        evaluators[user_id] = EvaluationAgent(
            role=role,
            experience_level=exp
        )

    evaluator = evaluators[user_id]
    print(evaluator , "evaluator")

    # Evaluate only if a previous question exists
    if question_asked:
        current_topic = result.get("topic")
        evaluator.add_question_answer(question_asked, previous_answer, current_topic , user_id)

    # update latest question
    question_asked = result.get("question")

    return result

if __name__ == "__main__":
    app.run(port=5000, debug=True)
