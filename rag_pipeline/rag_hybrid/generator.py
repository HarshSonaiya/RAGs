from fastapi import Form
from rag_pipeline.rag_hybrid.model import create_pipeline, initialize_tokenizer
from rag_pipeline.rag_hybrid.retriever import extract_content_from_pdf, setup_retrievers
from langchain.chains import RetrievalQA
from typing import Dict

async def process_rag_pipeline(file: str, query: str = Form(None)) -> Dict:

    if file:
        docs = extract_content_from_pdf(file)
    else:
        return {'message': "Invalid File"}

    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = initialize_tokenizer(model_name)
    llm = create_pipeline(model_name, tokenizer)

    vectorstore_retriever, ensemble_retriever, bm25_retriever = setup_retrievers(docs, model_name)

    normal_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore_retriever
    )
    hybrid_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=ensemble_retriever
    )

    response1 = normal_chain.invoke(query)
    response2 = hybrid_chain.invoke(query)
    response3_docs = bm25_retriever.get_relevant_documents(query)
    response3 = " ".join([doc.page_content for doc in response3_docs])

    return {
        "sparse_response": response3,
        "dense_response": response1.get("result"),
        "hybrid_response": response2.get("result"),
        "ragas_comparison": "Comparison results here"
    }
