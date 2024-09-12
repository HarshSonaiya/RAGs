from logging_config import get_logger
from fastapi import FastAPI, File, UploadFile, Form
from rag_pipeline.Qdrant_hybrid_rag.indexing import index_documents
from rag_pipeline.Qdrant_hybrid_rag.generator import answer_query
# from rag_pipeline.evaluator import generate_testset
from typing import List
import shutil
import os

app = FastAPI()

# Initialize logger
logger = get_logger("server")

@app.post("/api/ragas")
async def ragas_endpoint(files: List[UploadFile] = File(...), query: str = Form(...)):
    try :
        for file in files:
            temp_file_path = f"data/raw/{file.filename}"
            logger.info(f"Received file: {file.filename} and query: {query}")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)


            index_documents(temp_file_path)
            # generate_testset(temp_file_path)

        hybrid_response = answer_query(query)
        # logger.info("RAG pipeline processing completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred while processing RAG pipeline: {e}")
        return {"error": str(e)}

    return {
        "dense_response": "no response",
        "hybrid_response": hybrid_response,
        "sparse_response": "no response",
        "ragas_comparison": "no response"
    }

