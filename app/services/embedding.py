import logging
from transformers import AutoTokenizer, AutoModel
import os
import faiss
import torch
from datetime import datetime
import json
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Konfigurasi model
MODEL_NAME = os.getenv("MODEL_NAME", "indolem/indobert-base-uncased")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
INDEX_PATH = os.getenv("INDEX_PATH", "data/faqs_index.faiss")
METADATA_PATH = os.getenv("METADATA_PATH", "data/faqs_metadata.json")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.6"))

# Singleton untuk model dan index
class EmbeddingModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.load_model()
            cls._instance.load_index()
        return cls._instance
    
    def load_model(self):
        logger.info(f"Loading model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def load_index(self):
        try:
            if os.path.exists(INDEX_PATH):
                logger.info(f"Loading FAISS index from {INDEX_PATH}")
                self.index = faiss.read_index(INDEX_PATH)
                
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            else:
                logger.info("No existing index found, creating new one")
                self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product for cosine similarity
                self.metadata = {"faqs": {}, "last_updated": None}
                self.save_index()
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.metadata = {"faqs": {}, "last_updated": None}
    
    def save_index(self):
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            faiss.write_index(self.index, INDEX_PATH)
            
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Index saved with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def generate_embedding(self, text):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token embedding as sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Normalize to unit length for cosine similarity
        faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def search(self, query, top_k=3):
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        # Search in index
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 if not enough results
                faq_id = int(self.metadata["faqs"][str(idx)]["id"])
                similarity = float(scores[0][i])
                results.append({
                    "id": faq_id,
                    "question": self.metadata["faqs"][str(idx)]["question"],
                    "answer": self.metadata["faqs"][str(idx)]["answer"],
                    "similarity": similarity
                })
        
        return results
    
    def add_or_update_faq(self, faq_id, question, answer=None):
        # Generate embedding for question
        question_embedding = self.generate_embedding(question)
        
        # Check if ID already exists
        existing_idx = None
        for idx, metadata in self.metadata["faqs"].items():
            if metadata["id"] == faq_id:
                existing_idx = int(idx)
                break
        
        if existing_idx is not None:
            # Update existing embedding
            self.index.remove_ids(np.array([existing_idx], dtype=np.int64))
            
        # Add new embedding
        idx = self.index.ntotal
        self.index.add(question_embedding)
        
        # Update metadata
        self.metadata["faqs"][str(idx)] = {
            "id": faq_id,
            "question": question,
            "answer": answer
        }
        
        # Save updated index
        self.save_index()
        
        return idx
    
    def delete_faq(self, faq_id):
        # Find the index for this FAQ ID
        idx_to_remove = None
        for idx, metadata in self.metadata["faqs"].items():
            if metadata["id"] == faq_id:
                idx_to_remove = int(idx)
                break
        
        if idx_to_remove is not None:
            # Remove from index
            self.index.remove_ids(np.array([idx_to_remove], dtype=np.int64))
            
            # Update metadata
            new_metadata = {"faqs": {}, "last_updated": None}
            
            # Rebuild metadata with correct indices
            for idx, metadata in self.metadata["faqs"].items():
                if int(idx) != idx_to_remove:
                    new_idx = int(idx)
                    if new_idx > idx_to_remove:
                        new_idx -= 1
                    new_metadata["faqs"][str(new_idx)] = metadata
            
            self.metadata = new_metadata
            self.save_index()
            return True
        
        return False

    # Tambahkan method untuk Cek Konsistensi Postgesql <=> FAISS
    def save_consistency_result(self, result):
        self.metadata["consistency_check"] = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        self.save_index()
