from logging_config import get_logger
from fastapi import FastAPI, File, UploadFile, Form
from rag_pipeline.Qdrant_hybrid_rag.indexing import index_documents
from rag_pipeline.Qdrant_hybrid_rag.generator import answer_query, answer_dense_query, answer_hyde_query
from rag_pipeline.evaluation import evaluate_response
from typing import List
import shutil
import os

app = FastAPI()

# Initialize logger
logger = get_logger("server")

@app.post("/api/rag")
async def ragas_endpoint(files: List[UploadFile] = File(...), query: str = Form(...)):
    try :
        temp_file_path= ""
        for file in files:
            temp_file_path = f"data/raw/{file.filename}"
            logger.info(f"Received file: {file.filename} and query: {query}")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)


            index_documents(temp_file_path)
        # dense_response, dense_retrieved = answer_dense_query(query)
        # dense_llm_eval, dense_retriever_eval = evaluate_response(dense_retrieved, query, dense_response)

        hybrid_response, retrieved = answer_query(query)
        logger.info("Hybrid RAG pipeline processing completed successfully.")

        hybrid_llm_eval, hybrid_retriever_eval = evaluate_response(retrieved, query, hybrid_response)
        logger.info("Hybrid Evaluation completed successfully.")

        hyde_response, hyde_retrieved = answer_hyde_query(query)
        logger.info("HyDE RAG pipeline processing completed successfully.")

        hyde_llm_eval, hyde_retriever_eval = evaluate_response(hyde_retrieved, query, hyde_response)
        logger.info("Hybrid Evaluation completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred while processing RAG pipeline: {e}")
        return {"error": str(e)}

    return {
        # "dense_response": dense_response,
        # "dense_llm_eval": dense_llm_eval,
        # "dense_retriever_eval": dense_retriever_eval,
        "hybrid_response": hybrid_response,
        "hybrid_llm_eval":  hybrid_llm_eval,
        "hybrid_retriever_eval": hybrid_retriever_eval,
        "hyde_response": hyde_response,
        "hyde_llm_eval":  hyde_llm_eval,
        "hyde_retriever_eval": hyde_retriever_eval
    }

