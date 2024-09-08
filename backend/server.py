from logging_config import get_logger
from fastapi import FastAPI, File, UploadFile, Form
from rag_pipeline.rag_hybrid.generator import process_rag_pipeline
import shutil
import os

app = FastAPI()

# Initialize logger
logger = get_logger("server")

@app.post("/api/ragas")
async def ragas_endpoint(file: UploadFile = File(...), query: str = Form(None)):
    try :
        temp_file_path = f"data/raw/{file.filename}"
        logger.info(f"Received file: {file.filename} and query: {query}")

       # Ensure the directory exists
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        result = await process_rag_pipeline(temp_file_path, query)
        logger.info("RAG pipeline processing completed successfully.")

        # Dummy responses for demonstration (replace with actual responses from RAG pipeline)
        dense_response = result.get("dense_response", "No dense retriever response")
        hybrid_response = result.get("hybrid_response", "No hybrid retriever response")
        sparse_response = result.get("sparse_response", "No hybrid retriever response")
        ragas_comparison = result.get("ragas_comparison", "No comparison data")

    except Exception as e:
        logger.exception(f"An error occurred while processing RAG pipeline: {e}")
        return {"error": str(e)}

    return {
        "dense_response": dense_response,
        "hybrid_response": hybrid_response,
        "sparse_response": sparse_response,
        "ragas_comparison": ragas_comparison
    }

