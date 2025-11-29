from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import PyPDF2
import os
import json
from dotenv import load_dotenv
import redis
from flask_cors import CORS   # << Added

# Initialize Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

from patternagent import generate_question_patterns

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "http://localhost:3000"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)
# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=api_key,
    temperature=0.7
)

# ==============================
# Helper: Extract text from PDF
# ==============================
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()


# ==============================
# 1️⃣ MpSetTopicsFromResume
# ==============================
prompt_template_resume = PromptTemplate(
    input_variables=["resume_text", "jd_text"],
    template="""
You are an intelligent resume analysis agent. 
You are given a resume (and optionally a job description). Extract:
1. Candidate's name (if found)
2. Total years of experience
3. Technical skills (e.g., Java, Spring Boot, Kafka, MySQL, etc.)
4. For each skill, provide key topics the candidate should be evaluated on based on their experience.

Return only valid JSON in this format:
{{
  "candidateName": "<string>",
  "experienceYears": <number>,
  "userId": "<string or null>",
  "skills": ["<skill1>", "<skill2>", ...],
  "topicsToEvaluate": {{
    "<skill>": ["<topic1>", "<topic2>", ...]
  }}
}}

Resume Text:
{resume_text}

Job Description (optional):
{jd_text}
"""
)


@app.route("/MpSetTopicsFromResume", methods=["POST"])
def settopicsfromresume():
    if "resume" not in request.files:
        return jsonify({"error": "resume file is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("jd", "")
    user_id = request.form.get("userId", "USR_" + os.urandom(3).hex())

    resume_text = extract_text_from_pdf(resume_file)

    # Use PromptTemplate.format to substitute variables
    prompt = prompt_template_resume.format(resume_text=resume_text, jd_text=jd_text or "N/A")

    messages = [
        SystemMessage(content="You are a professional AI agent for resume skill extraction."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)

    try:
        data = response.content.strip()
        start_idx = data.find("{")
        end_idx = data.rfind("}") + 1
        parsed_json = json.loads(data[start_idx:end_idx])
        parsed_json["userId"] = user_id
    except Exception:
        return jsonify({"error": "Failed to parse LLM response", "raw": response.content}), 500

    question_patterns = generate_question_patterns(parsed_json, llm)

    role_to_store = request.form.get("role", "")
    exp_to_store  = request.form.get("exp", "")

    payload = {
        "question": question_patterns,
        "role": role_to_store,
        "experience": exp_to_store,
        "candidateName": parsed_json.get("candidateName")
    }

    try:
        redis_client.set(user_id, json.dumps(payload))
        redis_client.expire(user_id, 24 * 60 * 60)
    except Exception as e:
        print("Redis error:", str(e))

    return question_patterns


# ==============================
# 2️⃣ MpSetTopicsFromInput
# ==============================
prompt_template_input = PromptTemplate(
    input_variables=["skills", "experience", "candidateName", "userId"],
    template="""
You are an expert technical mentor.

The user provides:
- Candidate name: {candidateName}
- User ID: {userId}
- Skills: {skills}
- Experience (in years): {experience}

Your task:
Based on the provided skills and experience, list all important technical topics that the candidate should prepare for interviews.

Rules:
- For Fresher → Include all beginner-level fundamentals for each skill.
- For 1 year → Include beginner + intermediate practical topics.
- For 2–4 years → Include intermediate + some advanced topics.
- For 5+ years → Include advanced + architecture-level concepts.
- Return only valid JSON (no explanations, no questions).

JSON Format:
{{
  "candidateName": "{candidateName}",
  "experienceYears": {experience},
  "userId": "{userId}",
  "skills": [{skills}],
  "topicsToEvaluate": {{
    "<skill>": ["<topic1>", "<topic2>", "<topic3>", ...]
  }}
}}
"""
    # note: no template_format given — default formatting is used; placeholders kept as {var}
)


@app.route("/MpSetTopicsFromInput", methods=["POST"])
def settopicsfrominput():
    data = request.get_json()
    required = ["skills", "experience", "candidateName", "userId"]
    if not data or not all(k in data for k in required):
        return jsonify({"error": f"Missing required fields: {', '.join(required)}"}), 400

    skills = ", ".join(data["skills"]) if isinstance(data["skills"], list) else data["skills"]
    experience = data["experience"]
    candidate_name = data["candidateName"]
    user_id = data["userId"]

    prompt = prompt_template_input.format(
        skills=skills,
        experience=experience,
        candidateName=candidate_name,
        userId=user_id
    )

    messages = [
        SystemMessage(content="You are a senior technical trainer and interview mentor."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    try:
        resp = response.content.strip()
        start_idx = resp.find("{")
        end_idx = resp.rfind("}") + 1
        parsed_json = json.loads(resp[start_idx:end_idx])
    except Exception:
        return jsonify({"error": "Failed to parse LLM response", "raw": response.content}), 500

    question_patterns = generate_question_patterns(parsed_json, llm)

    def _normalize_experience(v):
        try:
            if isinstance(v, (int, float)):
                return int(round(v))
            if isinstance(v, str):
                import re
                m = re.search(r"(\d+)", v)
                if m:
                    return int(m.group(1))
        except:
            pass
        return None

    def _infer_role(parsed):
        role = parsed.get("role") or parsed.get("desiredRole")
        if role:
            return role
        skills = parsed.get("skills") or []
        if isinstance(skills, list) and len(skills) > 0:
            return f"{skills[0]} Developer"
        if isinstance(skills, str) and skills:
            return f"{skills} Developer"
        return "Developer"

    role_to_store = _infer_role(parsed_json)
    exp_to_store = _normalize_experience(parsed_json.get("experienceYears") or parsed_json.get("experience"))

    payload = {
        "question": question_patterns,
        "role": role_to_store,
        "experience": exp_to_store,
        "candidateName": parsed_json.get("candidateName")
    }

    try:
        redis_client.set(user_id, json.dumps(payload))
        redis_client.expire(user_id, 24 * 60 * 60)
        print("stored in redis successfully")
    except Exception as e:
        print("Redis error:", str(e))

    return question_patterns


# ==============================
# Run the App
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8085, debug=False)
