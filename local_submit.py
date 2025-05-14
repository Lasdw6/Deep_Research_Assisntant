import json
from smolagents import CodeAgent, HfApiModel
from tools.final_answer import FinalAnswerTool
import yaml
from dotenv import load_dotenv
import os
from models.openai_model import OpenAIModel

load_dotenv()

with open("metadata.jsonl", "r") as f:
    questions = [json.loads(line) for line in f]

model = OpenAIModel()

results = []
for q in questions[:20]:
    print(f"\nTask ID: {q['task_id']}")
    print(f"Question: {q['Question']}")
    try:
        answer = model.run(q["Question"])
    except Exception as e:
        print("Error:", e)
        answer = "error"
    results.append({
        "task_id": q["task_id"],
        "model_answer": str(answer).strip(),
        "reasoning_trace": None
    })

with open("submission.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
