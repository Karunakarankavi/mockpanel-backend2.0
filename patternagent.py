from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import json

prompt_template_pattern = PromptTemplate(
    input_variables=["topics_json", "experience"],
    template="""
You are a professional interview pattern designer.

You are given a list of skills and topics, and the candidate's experience.

For each topic, suggest 2–4 types of *question patterns* that an interviewer can use later 
to generate real questions dynamically.

Examples of question patterns: 
- Definition-based
- Comparison-based
- Scenario-based
- Code-based
- Configuration-based
- Optimization-based
- Troubleshooting-based
- Real-world usage-based
- Security-based

Rules:
1. For every topic, always include "Definition-based" as the first pattern.
2. For the remaining patterns, analyze the topic name and generate 1–2 additional relevant question patterns.
3.Adjust patterns to match the candidate's experience level:
 - 0–1 years: Conceptual & basic understanding patterns.
 - 1–2 years: Scenario-based and applied patterns.
 - 3+ years: Design, optimization, and troubleshooting patterns.



Return valid JSON like this:

{{
  "questionPatterns": {{
    "<skill>": {{
      "<topic>": ["<pattern1>", "<pattern2>", "<pattern3>"]
    }}
  }}
}}

Topics:
{topics_json}

Experience: {experience}
"""
)


def generate_question_patterns(parsed_topics_json, llm):
    """
    Takes the JSON output from MpSetTopicsFromInput and LLM instance,
    generates question patterns for each topic.
    """
    topics_dict = parsed_topics_json.get("topicsToEvaluate", {})
    experience = parsed_topics_json.get("experienceYears", 1)  # default 1 year if missing
    topics_json = json.dumps(topics_dict, indent=2)

    # Prepare prompt
    prompt = prompt_template_pattern.format(
        topics_json=topics_json,
        experience=experience
    )

    messages = [
        SystemMessage(content="You are a professional AI interviewer assistant."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    try:
        content = response.content.strip()
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        parsed_json = json.loads(content[start_idx:end_idx])
    except Exception:
        raise ValueError(f"Failed to parse question patterns: {response.content}")

    return parsed_json.get("questionPatterns", {})
