import logging
from logging_config import get_logger
from fastapi import Form
from rag_pipeline.rag_hybrid.model import create_pipeline, initialize_tokenizer
from rag_pipeline.rag_hybrid.retriever import extract_content_from_pdf, setup_retrievers
from langchain.chains import RetrievalQA
from typing import Dict

#Initialize logger
logger = get_logger("pipeline")

async def process_rag_pipeline(file: str, query: str = Form(None)) -> Dict:

    logger.info(f"Starting RAG pipeline for file: {file} with query: {query}")

    if file:
        docs = extract_content_from_pdf(file)
    else:
        logger.warning("No documents extracted from PDF.")
        return {'message': "Invalid File"}

    model_name = "HuggingFaceH4/zephyr-7b-beta"
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    tokenizer = initialize_tokenizer(model_name)
    llm = create_pipeline(model_name, tokenizer)

    logger.info("Tokenizer and llm initialized successfully.")

    vectorstore_retriever, ensemble_retriever, bm25_retriever = setup_retrievers(docs, embedding_model_name)

    logger.info("Retrievers created successfully.")

    normal_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore_retriever
    )
    logger.info("Normal Chain generated successfully.")

    hybrid_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=ensemble_retriever
    )
    logger.info("Hybrid Chain generated successfully.")

    bm25_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=bm25_retriever
    )
    logger.info("BM25 Chain generated successfully.")

    response1 = normal_chain.invoke(query)
    response2 = hybrid_chain.invoke(query)
    response3 = bm25_chain.invoke(query)
    print(response3)

    logger.info("RAG pipeline responses generated successfully.")

    return {
        "sparse_response": response3,
        "dense_response": response1.get("result"),
        "hybrid_response": response2.get("result"),
        "ragas_comparison": "Comparison results here"
    }
