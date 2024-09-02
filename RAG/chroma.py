import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import python-dotenv

HF_TOKEN = os.environ.get("HF_TOKEN")
# global default
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
# Settings.llm = HuggingFaceInferenceAPI(
#     model_name="CohereForAI/c4ai-command-r-plus", token=HF_TOKEN
# )
Settings.llm = Ollama(model="dolphin-phi", request_timeout=60.0)


def load_new_data():
    # load some documents
    print("loading data...")
    documents = SimpleDirectoryReader("./RAG/data").load_data()
    # initialize client, setting path to save data

    db = chromadb.PersistentClient(path="./RAG/ChromaStore")

    # create collection
    print("creating collection...")
    chroma_collection = db.get_or_create_collection("mycollection")

    print("creating vector store...")
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("creating index...")
    # # create your index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    return "Data loading successfully!"

def query_from_disk(query):
        # global default
    db = chromadb.PersistentClient(path="./RAG/ChromaStore")
    chroma_collection = db.get_or_create_collection("mycollection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store
    )
    print("configure retriver")
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    # #  # create a query engine and query
    print('querying')
    query_engine = index.as_query_engine(llm=Settings.llm, )
    print('querying')
    response = query_engine.query(query)
    
    return response


if __name__ == "__main__":
    # outcome = load_new_data()
    # print(outcome)
    response = query_from_disk("Which empirical approaches adopt multiobjecting optimization based methods?")
    print(response)
