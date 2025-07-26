from dotenv import load_dotenv
import os
from mistralai import Mistral
import yaml

load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

with open("./finetuned_llm/job_id.yaml", "r") as f:
    data = yaml.safe_load(f)
    job_id=data["job_id"]

def inference(job_id, messages):
    chat_response = client.chat.complete(
    model = client.fine_tuning.jobs.get(job_id=job_id).fine_tuned_model,
    messages = messages
    )
    return chat_response.choices[0].message.content if chat_response.choices[0].message.content else "No response from the model"


def run_inference(query):
    messages = [
            {"role": "user", "content": f"{query}"}
        ]
    return inference(job_id, messages)


if __name__ == "__main__":
    print(run_inference("What is the dining options at WPI?"))