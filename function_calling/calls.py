import numpy as np
import pandas as pd
import os
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



class VectorDatabase():
    def __init__(self):
        db_connection = os.environ.get('VDB_CONNECTION_DEV')
        # create or get a collection with 1536 dimensional vectors (default dimension for text-embedding-ada-002)
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.embedding_model = os.environ.get("EMBEDDING_MODEL")
        self.client = OpenAI(api_key=self.api_key)
        self.industry_embedding_path = os.environ.get("FUNCTION_EMBEDDING_PATH")

    def createEmbedding(self, sentence):
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=sentence
        )
        embedding = (sentence, response.data[0].embedding, {})
        return embedding

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)

        # Linear algebra normalise
        norm_a = np.linalg.norm(embedding1)
        norm_b = np.linalg.norm(embedding2)
        return dot_product / (norm_a * norm_b)

    def industry_segmentation(self, sentence):

        df = pd.read_csv(self.industry_embedding_path)
        embedding1 = self.createEmbedding(sentence)[1]
        index = 0
        industry = None
        for i in range(len(self.industry_list)):
            similarity = self.calculate_similarity(embedding1,list(df[self.industry_list[i]]))
            if index < similarity:
                index = similarity
                industry = self.industry_list[i]

        return industry
    
if __name__ == "__main__":
    # vdb = VectorDatabase()
    # sentence = "I am a software engineer"
    # print(vdb.industry_segmentation(sentence))

    ret = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5").aget_text_embedding("hello world")
    print(ret)