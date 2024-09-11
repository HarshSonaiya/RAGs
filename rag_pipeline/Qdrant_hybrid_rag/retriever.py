from fastapi import Form
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from logging_config import get_logger
from typing import Dict
from rag_pipeline.rag_hybrid.retriever import extract_content_from_pdf
from langchain.chains import ConversationalRetrievalChain,LLMChain
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from transformers import AutoModelForMaskedLM, AutoTokenizer


groq_api_key = <your-api-key>

logger = get_logger("pipeline")
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

# Initialize QdrantClient with Docker URL setup


model = SparseTextEmbedding(
    model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
)

tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
dense_embedding = AutoModelForMaskedLM.from_pretrained(embedding_model_name)
# delete collection if it exists
if client.collection_exists("rag"):
    client.delete_collection("rag")

vector_store = QdrantVectorStore(
    collection_name="llama2_bm42",
    client=client,
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
)

async def process_rag_pipeline(file: str, query: str = Form(None)) -> Dict:

    logger.info(f"Starting RAG pipeline for file: {file} with query: {query}")

    if file:
        documents = extract_content_from_pdf(file)
    else:
        logger.warning("No documents extracted from PDF.")
        return {'message': "Invalid File"}

    vectorstore = Qdrant.from_documents(documents=documents,
                                        client=client,
                                        aclient=aclient,
                                        enable_hybrid=True,
                                        batch_size=20,
                                        url= qdrant_url,
                                        prefer_grpc= False,
                                        collection_name="rag",)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )
    for doc in retriever:
        print(doc)

    logger.info("Retrievers created successfully.")

    template = """You are an AI assistant for answering questions about the various documents from the user.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:"""

    prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key= groq_api_key)

    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    question_generator_chain = LLMChain(llm=llm, prompt=prompt)

    qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=doc_chain,
    )

    dense_response = qa_chain.invoke({
        "question": query,  # line 533
        "chat_history": []
    })["answer"]

    # hybrid search in action
    filter_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1, "filter": {"source": file} }
    )
    filter_chain = ConversationalRetrievalChain(
        retriever=filter_retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=doc_chain,
    )
    hybrid_response = filter_chain.invoke({
        "question": query,
        "chat_history": [],
        "filter": filter,
    })["answer"]
    logger.info("Q&A Chain generated successfully.")

    # answers = []
    # contexts = []

    # for chain in chains:
    #     chain_results = []
    #     for question in questions:
    #         response = await chain.invoke(question)
    #         chain_results.append((question, response))
    #         contexts.append([docs.page_content for docs in chain.get_relevant_documents(question)])
    #
    # #     results.append(chain_results)
    # for question in questions:
    #     answers.append(hybrid_chain.invoke(question))
    #     contexts.append([docs.page_content for docs in ensemble_retriever.invoke(question)])
    #
    # # Preparing the dataset
    # data = {
    #     "question": questions,
    #     "answer": answers,
    #     "contexts": contexts,
    #     "ground_truth": ground_truth
    # }
    #
    # dataset = Dataset.from_dict(data)
    #
    # result = evaluate(
    #     dataset=dataset,
    #     metrics=[
    #         context_precision,
    #         context_recall,
    #         faithfulness,
    #         answer_relevancy,
    #     ],
    # )
    # #
    # # response = []
    # # for i, chain_results in enumerate(results):
    # #     dataset = dataset
    # #     result = evaluate(
    # #         dataset=dataset,
    # #         metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    # #     )
    # #     print(f"RAG Chain {i + 1} Results:")
    # #     print(result)
    # #     response.append(result)
    #
    # logger.info("RAGAs evaluation completed successfully.")
    #
    # response4 = result
    #
    logger.info("RAG pipeline responses generated successfully.")

    return {
        "sparse_response": "No response",
        "dense_response": dense_response,
        "hybrid_response": hybrid_response,
        "ragas_comparison": "No response"
    }