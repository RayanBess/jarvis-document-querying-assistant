import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import openai
from llama_index.llms.openai import OpenAI
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
import requests

os.environ["OPENAI_API_KEY"] = "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"
HF_TOKEN = "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"
# global default
Settings.embed_model = HuggingFaceInferenceAPI(
    model_name="BAAI/bge-small-en-v1.5",


)
Settings.llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha", token=HF_TOKEN
)


def load_new_data():
    # load some documents
    documents = SimpleDirectoryReader("./data").load_data()
    # initialize client, setting path to save data

    db = chromadb.PersistentClient(path="./ChromaStore")

    # create collection
    chroma_collection = db.get_or_create_collection("mycollection")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # # create your index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    return "success"

def query_from_disk(query):
        # global default
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    db = chromadb.PersistentClient(path="./ChromaStore")
    chroma_collection = db.get_or_create_collection("mycollection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store
    )
    # #  # create a query engine and query
    print('querying')
    query_engine = index.as_query_engine(llm = "")
    response = query_engine.query(query)
    
    return response



if __name__ == "__main__":
    # load_new_data()
    response = query_from_disk("How is AI affecting cybersecurity?")
    print(response)
