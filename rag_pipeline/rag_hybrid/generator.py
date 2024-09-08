from fastapi import Form
from rag_pipeline.rag_hybrid.model import create_pipeline, initialize_tokenizer
from rag_pipeline.rag_hybrid.retriever import extract_content_from_pdf, setup_retrievers
from langchain.chains import RetrievalQA
from typing import Dict
from logging_config import get_logger
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


#Initialize logger
logger = get_logger("pipeline")


questions = [
    "What are LSTMS ?",
    "What is the role of Cell-State in LSTMS ?"
]

ground_truth = [
    "Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) designed to model sequential data and capture long-term dependencies.",
    "The cell state in LSTMs acts as a memory component that carries information across the sequence.",
]

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

    # response1 = normal_chain.invoke(query)
    # response2 = hybrid_chain.invoke(query)
    # response3 = bm25_chain.invoke(query)

    answers = []
    contexts = []

    # for chain in chains:
    #     chain_results = []
    #     for question in questions:
    #         response = await chain.invoke(question)
    #         chain_results.append((question, response))
    #         contexts.append([docs.page_content for docs in chain.get_relevant_documents(question)])
    #
    #     results.append(chain_results)
    for question in questions:
        answers.append(hybrid_chain.invoke(question))
        contexts.append([docs.page_content for docs in ensemble_retriever.invoke(question)])

    # Preparing the dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )
    #
    # response = []
    # for i, chain_results in enumerate(results):
    #     dataset = dataset
    #     result = evaluate(
    #         dataset=dataset,
    #         metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    #     )
    #     print(f"RAG Chain {i + 1} Results:")
    #     print(result)
    #     response.append(result)

    logger.info("RAGAs evaluation completed successfully.")

    response4 = result

    logger.info("RAG pipeline responses generated successfully.")

    return {
        # "sparse_response": response3.get("result"),
        # "dense_response": response1.get("result"),
        # "hybrid_response": response2.get("result"),
        "ragas_comparison": response4
    }



#
# async def test_set_generation(model, embedding_model, documents):
#     generator_llm = model
#     critic_llm = model
#     embeddings = embedding_model
#
#     generator = TestsetGenerator.from_langchain(
#         generator_llm,
#         critic_llm,
#         embeddings
#     )
#
#     # Change resulting question type distribution
#     distributions = {
#         simple: 0.5,
#         multi_context: 0.4,
#         reasoning: 0.1
#     }
#
#     test_set = generator.generate_with_langchain_docs(documents, 10, distributions)
#     return test_set.to_pandas()
#
#
# async def evaluation(model, embedding_model, docs, chains):
#
#     return result.to_pandas()
