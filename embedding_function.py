
import ollama
import os
from chromadb import EmbeddingFunction


# Make it a LangChain-style embedding function
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model=os.getenv('EMBEDDING_MODEL')):
        self.model = model
        
    def _call_ollama_single(self, input_text):
        resp = ollama.embeddings(model=self.model, prompt=input_text)
        return resp["embedding"]  

    def embed_documents(self, input_texts):
        return [self._call_ollama_single(input_text) for input_text in input_texts]
    
    def embed_query(self, input_text):
        return self._call_ollama_single(input_text)