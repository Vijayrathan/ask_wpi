from mistralai import Mistral
import os
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)


def upload_dataset():    
    dataset = client.files.upload(file={
        "file_name": "dataset.jsonl",
        "content": open("./finetuned_llm/dataset.jsonl", "rb"),
    })
    return dataset.id

def create_job(file_id):
    created_jobs = client.fine_tuning.jobs.create(
    model="open-mistral-7b", 
    training_files=[{"file_id":f"{file_id}", "weight": 1}],
    hyperparameters={
        "training_steps": 10,
        "learning_rate":0.0001
    },
    auto_start=False
    )
    return created_jobs.id

def initialize_job(job_id):
    client.fine_tuning.jobs.start(job_id=job_id)
    return job_status(job_id)

def job_status(job_id):
    while True:
        jobs = client.fine_tuning.jobs.get(job_id=job_id)
        print(jobs)
        if jobs.status == "VALIDATED":
            return "VALIDATED"
        elif jobs.status == "FAILED":
            raise Exception("Job validation failed!")
        time.sleep(10) 
def inference(job_id, messages):
    chat_response = client.chat.complete(
    model = client.fine_tuning.jobs.get(job_id=job_id).fine_tuned_model,
    messages = messages
    )
    return chat_response



if __name__ == "__main__":
    # file_id = upload_dataset()
    # job_id = create_job(file_id)
    # print(f"Job ID: {job_id}")
    # print(f"Job Status: {initialize_job(job_id)}")
    # print(f"File ID: {file_id}")
    job_id = "226cd2e1-6352-465e-b0eb-0c47fedc4fdd"
    print(client.fine_tuning.jobs.get(job_id=job_id).fine_tuned_model)
    messages = [
        {"role": "user", "content": "Where is WPI located?"}
    ]
    print(inference(job_id, messages))