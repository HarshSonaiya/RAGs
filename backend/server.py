from logging_config import get_logger
from fastapi import FastAPI, File, UploadFile, Form
from rag_pipeline.Qdrant_hybrid_rag.generator import answer_query, create_llm_chain
from rag_pipeline.Qdrant_hybrid_rag.retriever import process_retrievers
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

        retrievers = await process_retrievers(temp_file_path, query)
        sparse_retriever = retrievers["sparse_retriever"]
        dense_retriever = retrievers["dense_retriever"]
        hybrid_retriever = retrievers["hybrid_retriever"]

        llm_chain = create_llm_chain()

        sparse_response = await answer_query(llm_chain, sparse_retriever, query)
        dense_response = await answer_query(llm_chain, dense_retriever, query)
        hybrid_response = await answer_query(llm_chain, hybrid_retriever, query)

        logger.info("RAG pipeline processing completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred while processing RAG pipeline: {e}")
        return {"error": str(e)}

    return {
        "dense_response": dense_response,
        "hybrid_response": hybrid_response,
        "sparse_response": sparse_response,
        "ragas_comparison": ragas_comparison
    }

