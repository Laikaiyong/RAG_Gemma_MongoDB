
import streamlit as st
from datetime import datetime
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
import json
import boto3
from botocore.config import Config
import os
from typing import List, Dict
from pymongo import MongoClient

# Name of the database -- Change if needed or leave as is
DB_NAME = "mongodb_rag_lab"
# Name of the collection -- Change if needed or leave as is
COLLECTION_NAME = "knowledge_base"
# Name of the vector search index -- Change if needed or leave as is
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"


rerank_model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
    
mongodb_client = MongoClient(st.secrets["mongo"]["host"], appname="devrel.workshop.rag")
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

history_collection = mongodb_client[DB_NAME]["chat_history"]

my_config = Config(
    region_name = 'us-east-1',
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)


def store_chat_message(session_id: str, role: str, content: str) -> None:
    """
    Store a chat message in a MongoDB collection.

    Args:
        session_id (str): Session ID of the message.
        role (str): Role for the message. One of `system`, `user` or `assistant`.
        content (str): Content of the message.
    """
    # Create a message object with `session_id`, `role`, `content` and `timestamp` fields
    # `timestamp` should be set the current timestamp
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(),
    }
    # Insert the `message` into the `history_collection` collection
    history_collection.insert_one(message)


class TitanEmbeddings(object):
    accept = "application/json"
    content_type = "application/json"

    def __init__(self, model_id="amazon.titan-embed-text-v2:0"):
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            config=my_config,
            aws_access_key_id=st.secrets["aws"]["access_key"],
            aws_secret_access_key=st.secrets["aws"]["secret_key"]
        )
        self.model_id = model_id
    def __call__(self, text, dimensions, normalize=True):
        """
        Returns Titan Embeddings
        Args:
            text (str): text to embed
            dimensions (int): Number of output dimensions.
            normalize (bool): Whether to return the normalized embedding or not.
        Return:
            List[float]: Embedding

        """
        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        })
        response = self.bedrock.invoke_model(
            body=body, modelId=self.model_id, accept=self.accept, contentType=self.content_type
        )
        response_body = json.loads(response.get('body').read())
        return response_body['embedding']



def get_embedding(text: str) -> List[float]:
    """
    Generate the embedding for a piece of text.

    Args:
        text (str): Text to embed.

    Returns:
        List[float]: Embedding of the text as a list.
    """
    embedding_model = TitanEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    dimensions = 1024
    normalize = True
    embedding = embedding_model(text, dimensions, normalize)

    return embedding



def vector_search(user_query: str) -> List[Dict]:
    """
    Retrieve relevant documents for a user query using vector search.

    Args:
    user_query (str): The user's query string.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the `user_query` using the `get_embedding` function defined in Step 5
    query_embedding = get_embedding(user_query)

    # Define an aggregation pipeline consisting of a $vectorSearch stage, followed by a $project stage
    # Set the number of candidates to 150 and only return the top 5 documents from the vector search
    # In the $project stage, exclude the `_id` field and include only the `body` field and `vectorSearchScore`
    # NOTE: Use variables defined previously for the `index`, `queryVector` and `path` fields in the $vectorSearch stage
    pipeline = [
      {
          "$vectorSearch": {
              "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
              "queryVector": query_embedding,
              "path": "embedding",
              "numCandidates": 150,
              "limit": 5,
          }
      },
      {
          "$project": {
              "_id": 0,
              "Content": 1,
              "score": {"$meta": "vectorSearchScore"}
          }
      }
  ]

    # Execute the aggregation `pipeline` and store the results in `results`
    results = collection.aggregate(pipeline)

    return list(results)



def create_prompt(user_query: str) -> str:
    """
    Create a chat prompt that includes the user query and retrieved context.

    Args:
        user_query (str): The user's query string.

    Returns:
        str: The chat prompt string.
    """
    # Retrieve the most relevant documents for the `user_query` using the `vector_search` function defined in Step 8
    context = vector_search(user_query)
    # Extract the "body" field from each document in `context`
    documents = [d.get("Content") for d in context]
    # Use the `rerank_model` instantiated above to re-rank `documents`
    # Set the `top_k` argument to 5
    reranked_documents = rerank_model.rank(
        user_query, documents, return_documents=True, top_k=5
    )
    # Join the re-ranked documents into a single string, where each document is separated by two new lines ("\n\n")
    context = "\n\n".join([d.get("text", "") for d in reranked_documents])
    # Prompt consisting of the question and relevant context to answer it
    prompt = f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}\n\nQuestion:{user_query}"
    return prompt




def retrieve_session_history(session_id: str) -> List:
    """
    Retrieve chat message history for a particular session.

    Args:
        session_id (str): Session ID to retrieve chat message history for.

    Returns:
        List: List of chat messages.
    """
    # Query the `history_collection` collection for documents where the "session_id" field has the value of the input `session_id`
    # Sort the results in increasing order of the values in `timestamp` field
    cursor =  history_collection.find({"session_id": session_id}).sort("timestamp", 1)

    if cursor:
        # Iterate through the cursor and extract the `role` and `content` field from each entry
        # Then format each entry as: {"role": <role_value>, "content": <content_value>}
        messages = [{"role": msg["role"], "content": [{ "text": msg["content"]}]} for msg in cursor]
    else:
        # If cursor is empty, return an empty list
        messages = []

    return messages




def generate_answer(session_id: str, user_query: str) -> None:
    """
    Generate an answer to the user's query taking chat history into account.

    Args:
        session_id (str): Session ID to retrieve chat history for.
        user_query (str): The user's query string.
    """
    # Initialize list of messages to pass to the chat completion model
    messages = []

    # Retrieve documents relevant to the user query and convert them to a single string
    context = vector_search(user_query)
    context = "\n\n".join([d.get("Content", "") for d in context])

    # # Create a system prompt containing the retrieved context
    # system_message = {
    #     "role": "assistant",
    #     "content": [
    #         {
    #             "text": f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}",
    #         }
    #     ]
    # }
    # # Append the system prompt to the `messages` list
    # messages.append(system_message)

    # Use the `retrieve_session_history` function to retrieve message history from MongoDB for the session ID `session_id`
    # And add all messages in the message history to the `messages` list
    message_history = retrieve_session_history(session_id)

    messages.extend(message_history)
    prompt = create_prompt(user_query)

    # Format the user message in the format {"role": <role_value>, "content": <content_value>}
    # The role value for user messages must be "user"
    # And append the user message to the `messages` list
    user_message = {"role": "user", "content": [{ "text": prompt}]}


    messages.append(user_message)

    # Call the chat completions API
    client = boto3.client(
          service_name='bedrock-runtime',
          config=my_config,
          aws_access_key_id=st.secrets["aws"]["access_key"],
          aws_secret_access_key=st.secrets["aws"]["secret_key"]
    )

    # Set the model ID, e.g., Titan Text Premier.
    model_id = "amazon.titan-text-premier-v1:0"

    # Use the `prompt` created above to populate the `content` field in the chat message
    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"topP":0.9},
        additionalModelRequestFields={}
    )

    # Extract the answer from the API response
    answer = response["output"]["message"]["content"][0]["text"]

    # Use the `store_chat_message` function to store the user message and also the generated answer in the message history collection
    # The role value for user messages is "user", and "assistant" for the generated answer
    store_chat_message(session_id, "user", user_query)
    store_chat_message(session_id, "assistant", answer)

    return answer

