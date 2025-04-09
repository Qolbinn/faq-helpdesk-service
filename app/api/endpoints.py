from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.models.schemas import FAQItem, QueryInput, QueryResponse
from app.services.embedding import EmbeddingModel
import os
from typing import List, Optional, Dict, Any
import logging
import json
from datetime import datetime
import faiss

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
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.8"))

router = APIRouter()

# Dependency to get model instance
def get_embedding_model():
    return EmbeddingModel()



# MAIN API ROUTE
@router.get("/")
def read_root():
    return {"message": "FAQ Chatbot API is running"}

# Tambahkan route lainnya
@router.get("/status")
def get_status(model: EmbeddingModel = Depends(get_embedding_model)):
    return {
        "status": "online",
        "model": MODEL_NAME,
        "faqs_count": model.index.ntotal,
        "last_updated": model.metadata.get("last_updated", None)
    }

# @router.post("/query", response_model=QueryResponse)
# def query(input_data: QueryInput, model: EmbeddingModel = Depends(get_embedding_model)):
#     results = model.search(input_data.query, input_data.top_k)
    
#     # Filter by threshold
#     filtered_results = [r for r in results if r["similarity"] >= input_data.threshold]

#     # # Menghapus answer dari alternative results
#     # formatted_results = [
#     #     {key: value for key, value in r.items() if key != "answer"} 
#     #     for r in filtered_results
#     # ]

#     # Format response
#     response = {"results": filtered_results}
    
#     # Add best match if available
#     if filtered_results:
#         response["best_match"] = filtered_results[0]
    
#     return response

@router.post("/query", response_model=dict)
def query(input_data: QueryInput, model: EmbeddingModel = Depends(get_embedding_model)):
    results = model.search(input_data.query, input_data.top_k)
    
    # Filter by threshold
    filtered_results = [r for r in results if r["similarity"] >= input_data.threshold]

    # Tentukan best match
    best_match = filtered_results[0] if filtered_results else None

    # Jika best_match memiliki similarity > HIGH_CONFIDENCE_THRESHOLD, tampilkan answer-nya
    if best_match and best_match["similarity"] > HIGH_CONFIDENCE_THRESHOLD:
        return {
            "message": "Best match found",
            "best_match": {
                "question": best_match["question"],
                "answer": best_match["answer"],
                "similarity": best_match["similarity"]
            }
        }
    
    # Jika similarity <= 0.8, tampilkan daftar alternatif (tanpa answer)
    alternative_questions = [
        {"id": r["id"], "question": r["question"], "similarity": r["similarity"]}
        for r in filtered_results
    ]
    return {
        "message": "No high-confidence match found. Please select from the alternatives.",
        "alternatives": alternative_questions
    }

@router.post("/faqs", status_code=201)
def create_faq(faq: FAQItem, model: EmbeddingModel = Depends(get_embedding_model)):
    idx = model.add_or_update_faq(faq.id, faq.question, faq.answer)
    return {"id": faq.id, "index": idx, "message": "FAQ created successfully"}

@router.put("/faqs/{faq_id}")
def update_faq(faq_id: int, faq: FAQItem, model: EmbeddingModel = Depends(get_embedding_model)):
    if faq.id != faq_id:
        raise HTTPException(status_code=400, detail="FAQ ID in path does not match body")
    
    idx = model.add_or_update_faq(faq.id, faq.question, faq.answer)
    return {"id": faq.id, "index": idx, "message": "FAQ updated successfully"}

@router.delete("/faqs/{faq_id}")
def delete_faq(faq_id: int, model: EmbeddingModel = Depends(get_embedding_model)):
    success = model.delete_faq(faq_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"FAQ with ID {faq_id} not found")
    
    return {"id": faq_id, "message": "FAQ deleted successfully"}

@router.post("/bulk_index")
def bulk_index(faqs: List[FAQItem], background_tasks: BackgroundTasks, model: EmbeddingModel = Depends(get_embedding_model)):
    def process_bulk_indexing(items):
        for faq in items:
            model.add_or_update_faq(faq.id, faq.question, faq.answer)
        logger.info(f"Bulk indexing of {len(items)} FAQs completed")
    
    # Process in background to avoid timeout for large datasets
    background_tasks.add_task(process_bulk_indexing, faqs)
    
    return {"message": f"Bulk indexing of {len(faqs)} FAQs initiated", "status": "processing"}




# ROUTE for FAISS

@router.get("/faiss/stats", response_model=dict)
def get_faiss_stats(model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Mendapatkan statistik tentang indeks FAISS
    """
    # Hitung jumlah FAQ per ID
    faq_count_by_id = {}
    for idx, metadata in model.metadata["faqs"].items():
        faq_id = metadata["id"]
        if faq_id in faq_count_by_id:
            faq_count_by_id[faq_id] += 1
        else:
            faq_count_by_id[faq_id] = 1
    
    # Cari duplikat
    duplicates = {faq_id: count for faq_id, count in faq_count_by_id.items() if count > 1}
    
    return {
        "total_vectors": model.index.ntotal,
        "dimension": model.index.d,
        "index_type": type(model.index).__name__,
        "last_updated": model.metadata.get("last_updated", None),
        "unique_faq_ids": len(faq_count_by_id),
        "has_duplicates": len(duplicates) > 0,
        "duplicates": duplicates,
        "memory_usage_mb": round(model.index.ntotal * model.index.d * 4 / (1024 * 1024), 2)  # Estimasi kasar
    }

@router.get("/faiss/list", response_model=list)
def list_faiss_items(
    skip: int = 0, 
    limit: int = 100,
    model: EmbeddingModel = Depends(get_embedding_model)
):
    """
    Mendapatkan daftar item dalam indeks FAISS
    """
    items = []
    
    # Ambil semua key dan urutkan berdasarkan indeks numerik
    sorted_keys = sorted([int(k) for k in model.metadata["faqs"].keys()])
    
    # Batasi dengan parameter skip dan limit
    paginated_keys = sorted_keys[skip:skip + limit]
    
    for idx in paginated_keys:
        metadata = model.metadata["faqs"].get(str(idx))
        if metadata:
            items.append({
                "index": idx,
                "id": metadata["id"],
                "question": metadata["question"],
                "answer": metadata.get("answer", "")
            })
    
    return items

@router.delete("/faiss/reset", status_code=204)
def reset_faiss_index(model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Menghapus semua data dalam indeks FAISS dan memulai dari awal
    HATI-HATI: Operasi ini tidak dapat dibatalkan!
    """
    try:
        # Buat indeks baru dengan dimensi yang sama
        dimension = model.index.d
        new_index = faiss.IndexFlatIP(dimension)
        
        # Ganti indeks lama
        model.index = new_index
        model.metadata = {"faqs": {}, "last_updated": datetime.now().isoformat()}
        
        # Simpan indeks kosong
        model.save_index()
        
        return {}
    except Exception as e:
        logger.error(f"Error resetting FAISS index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset index: {str(e)}")

@router.post("/faiss/export")
def export_faiss_data(model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Mengekspor data dari indeks FAISS untuk backup
    """
    try:
        # Buat direktori untuk ekspor jika belum ada
        export_dir = os.path.join(os.path.dirname(INDEX_PATH), "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Buat nama file dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_file = os.path.join(export_dir, f"faqs_index_{timestamp}.faiss")
        metadata_file = os.path.join(export_dir, f"faqs_metadata_{timestamp}.json")
        
        # Simpan file
        faiss.write_index(model.index, index_file)
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(model.metadata, f, ensure_ascii=False, indent=2)
        
        return {
            "message": "FAISS data exported successfully",
            "files": {
                "index": index_file,
                "metadata": metadata_file
            }
        }
    except Exception as e:
        logger.error(f"Error exporting FAISS data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export FAISS data: {str(e)}")

@router.post("/faiss/backup", response_model=dict)
def backup_faiss_data(model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Membuat backup sederhana dari indeks FAISS saat ini
    """
    try:
        # Buat nama file backup dengan timestamp
        backup_dir = os.path.join(os.path.dirname(INDEX_PATH), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_index = os.path.join(backup_dir, f"faqs_index_{timestamp}.faiss")
        backup_metadata = os.path.join(backup_dir, f"faqs_metadata_{timestamp}.json")
        
        # Salin file saat ini ke backup
        faiss.write_index(model.index, backup_index)
        
        with open(backup_metadata, "w", encoding="utf-8") as f:
            json.dump(model.metadata, f, ensure_ascii=False, indent=2)
        
        # Tambahkan informasi backup ke metadata
        if "backups" not in model.metadata:
            model.metadata["backups"] = []
            
        model.metadata["backups"].append({
            "timestamp": timestamp,
            "index_file": backup_index,
            "metadata_file": backup_metadata
        })
        
        # Simpan metadata yang diupdate
        model.save_index()
        
        return {
            "message": "Backup created successfully",
            "timestamp": timestamp,
            "files": {
                "index": backup_index,
                "metadata": backup_metadata
            }
        }
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

@router.get("/faiss/similar-questions/{faq_id}", response_model=list)
def find_similar_questions(
    faq_id: int, 
    threshold: float = 0.7, 
    top_k: int = 5,
    model: EmbeddingModel = Depends(get_embedding_model)
):
    """
    Menemukan pertanyaan serupa untuk FAQ tertentu
    """
    # Cari pertanyaan dengan ID yang diberikan
    question = None
    for idx, metadata in model.metadata["faqs"].items():
        if metadata["id"] == faq_id:
            question = metadata["question"]
            break
    
    if not question:
        raise HTTPException(status_code=404, detail=f"FAQ with ID {faq_id} not found")
    
    # Cari pertanyaan serupa
    results = model.search(question, top_k + 1)  # +1 karena akan termasuk dirinya sendiri
    
    # Filter hasil untuk menghapus dirinya sendiri dan yang di bawah threshold
    filtered_results = [
        r for r in results 
        if r["id"] != faq_id and r["similarity"] >= threshold
    ]
    
    return filtered_results







# Cek Konsistensi Postgresql <=> FAISS
@router.post("/consistency-check", response_model=dict)
def check_and_reindex(background_tasks: BackgroundTasks, model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Memeriksa konsistensi antara ID FAQ di database dan FAISS index.
    Jika ada yang tidak konsisten, otomatis melakukan reindexing.
    """
    # Struktur untuk menyimpan hasil pengecekan
    result = {
        "status": "checking",
        "faqs_total": 0,
        "missing_in_faiss": [],
        "orphaned_in_faiss": []
    }
    
    # Fungsi untuk melakukan pengecekan dan reindexing
    def perform_check():
        # Ini adalah titik integrasi dengan Laravel
        # Dapatkan data dari API Laravel untuk mendapatkan semua FAQ
        try:
            # Gunakan httpx untuk memanggil API Laravel
            import httpx
            
            # Ganti URL ini dengan endpoint Laravel yang mengembalikan semua FAQ
            laravel_api_url = os.getenv("LARAVEL_API_URL", "http://localhost/api/internal/all-faqs")
            
            # Opsional: Tambahkan header autentikasi jika diperlukan
            headers = {
                "Authorization": f"Bearer {os.getenv('API_TOKEN', 'secret-token')}"
            }
            
            # Panggil API Laravel
            response = httpx.get(laravel_api_url, headers=headers, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to get FAQs from Laravel: {response.text}")
                return
            
            # Data FAQ dari PostgreSQL
            postgres_faqs = response.json()
            postgres_faq_ids = {faq["id"] for faq in postgres_faqs}
            
            # Data FAQ di FAISS
            faiss_faq_ids = set()
            for idx in model.metadata["faqs"]:
                faq_id = model.metadata["faqs"][idx]["id"]
                faiss_faq_ids.add(faq_id)
            
            # Cari FAQ yang ada di PostgreSQL tapi tidak ada di FAISS
            missing_in_faiss = postgres_faq_ids - faiss_faq_ids
            
            # Cari FAQ yang ada di FAISS tapi tidak ada di PostgreSQL
            orphaned_in_faiss = faiss_faq_ids - postgres_faq_ids
            
            # Update struktur hasil
            result["faqs_total"] = len(postgres_faqs)
            result["missing_in_faiss"] = list(missing_in_faiss)
            result["orphaned_in_faiss"] = list(orphaned_in_faiss)
            
            # Tambahkan FAQ yang hilang ke FAISS
            if missing_in_faiss:
                logger.info(f"Adding {len(missing_in_faiss)} missing FAQs to FAISS index")
                for faq in postgres_faqs:
                    if faq["id"] in missing_in_faiss:
                        model.add_or_update_faq(
                            faq["id"], 
                            faq["question"],
                            faq["answer"]
                        )
            
            # Hapus FAQ yang sudah tidak ada dari FAISS
            if orphaned_in_faiss:
                logger.info(f"Removing {len(orphaned_in_faiss)} orphaned FAQs from FAISS index")
                for faq_id in orphaned_in_faiss:
                    model.delete_faq(faq_id)
            
            # Update status akhir
            if missing_in_faiss or orphaned_in_faiss:
                result["status"] = "reindexed"
            else:
                result["status"] = "consistent"
                
            logger.info(f"Consistency check completed: {result}")
                
        except Exception as e:
            logger.error(f"Error during consistency check: {e}")
            result["status"] = "error"
            result["error"] = str(e)
    
    # Jalankan tugas di background untuk menghindari timeout
    background_tasks.add_task(perform_check)
    
    return {
        "message": "Consistency check and reindexing started",
        "status": "processing"
    }

@router.get("/consistency-status")
def get_consistency_status(model: EmbeddingModel = Depends(get_embedding_model)):
    """
    Mendapatkan status pengecekan konsistensi terakhir
    """
    # Cek jika ada hasil konsistensi yang disimpan dalam metadata
    if "consistency_check" in model.metadata:
        return model.metadata["consistency_check"]
    else:
        return {"status": "never_run", "message": "Consistency check has never been run"}