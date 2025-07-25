#!/opt/anaconda3/bin/python

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from dotenv import load_dotenv

from pprint import pprint

#######################
# AskWPI ReACT prompt #
#######################

class AskWPIReactPrompt:
    # prompt text source:  https://github.com/arunpshankar/react-from-scratch/blob/main/data/input/react.txt

    PROMPT = """
You are a ReAct (Reasoning and Acting) agent tasked with answering the following query:

Query: {query}

Your goal is to reason about the query and decide on the best course of action to answer it accurately.

Previous reasoning steps and observations: {history}

Available tools: askwpi

Instructions:
1. Analyze the query, previous reasoning steps, and observations.
2. Decide on the next action: use a tool or provide a final answer.
3. Respond in the following JSON format:

If you need to use a tool:
{{
    "thought": "Your detailed reasoning about what to do next",
    "action": {{
        "name": "Tool name (askwpi or none)",
        "reason": "Explanation of why you chose this tool",
        "input": "Specific input for the tool, if different from the original query"
    }}
}}

If you have enough information to answer the query:
{{
    "thought": "Your final reasoning process",
    "answer": "Your comprehensive answer to the query"
}}

Remember:
- Be thorough in your reasoning.
- Use tools when you need more information.
- Always base your reasoning on the actual observations from tool use.
- If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
- Provide a final answer only when you're confident you have sufficient information.
- If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.
"""

    def __init__(self, query):
        self.query = query
        self.history = ""

    def get_prompt(self):
        return self.PROMPT.format(query=self.query, history=self.history)

    def update_history(self, text):
        #print(f"history text: {text}")
        self.history += f" {text}"
        #print(f"updated history: {self.history}")

###############################################
# utility functions to talk to the DB and LLM #
###############################################

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def get_embedding(chunk):
    embedding=embedding_model.encode(chunk)
    return embedding

def retrieve(query):
    try:
        db_client=chromadb.PersistentClient(path='../chroma_db')
        collection=db_client.get_collection("wpi_docs")
        query_embedding = get_embedding(query)
        results=collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=["embeddings", "metadatas", "documents", "distances"]
        )
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []

###########################################
# AskWPI ReACT main think/decide/act loop #
###########################################

def run_react_loop(query):
    max_iterations = 5
    iteration = 0

    api_key = os.getenv("MISTRAL_API_KEY")
    model = "mistral-large-latest"

    llm_client = Mistral(api_key=api_key)

    # create the prompt
    askwpi_llm_prompt = AskWPIReactPrompt(query)
    #print(askwpi_llm_prompt.get_prompt())

    while iteration <= max_iterations:
        iteration += 1
        print(f"\n----> iteration number {iteration}")

        if iteration > max_iterations:
            return "Reached max interations."
        else:
            # query the LLM - ReACT THINK

            print(askwpi_llm_prompt.get_prompt())

            chat_messages = [
                {"role": "user", "content": askwpi_llm_prompt.get_prompt()}
            ]

            chat_response = llm_client.chat.complete(model=model, messages=chat_messages)
            response_json = chat_response.choices[0].message.content.removeprefix('```json\n')
            response_json = response_json.removesuffix('\n```')
            print(f"\n\nRaw Chat Response: {chat_response}")
            #print(f"\n\nChat Response: {response_json}")

            decoded_response = json.loads(response_json)
            print("\n\nResponse dict:\n")
            pprint(decoded_response)

            # ReACT DECIDE
            if 'action' in decoded_response:
                print(f"take action: {decoded_response['action']['name']}: {decoded_response['action']['input']}")
                # query the DB - ReACT ACT
                results=retrieve(decoded_response['action']['input'])
                pprint(results)
                askwpi_llm_prompt.update_history(" ".join(results['documents'][0]))
            elif 'answer' in decoded_response:
                return decoded_response['answer']
            else:
                return "Invalid LLM response format"

if __name__ == "__main__":
    load_dotenv()

    #query="Who is the dean of the college of engineering?"
    query="What are the dining options at WPI?"
    result = run_react_loop(query)
    print(f"\n\nResult\n------\n{result}\n")
