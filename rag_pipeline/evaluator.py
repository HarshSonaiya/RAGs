from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ragas.testset.evolutions import reasoning, simple, multi_context
from rag_pipeline.Qdrant_hybrid_rag.retriever import hybrid_search
from rag_pipeline.Qdrant_hybrid_rag.generator import initialize_llm,create_llm_chain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import load_dataset, Dataset
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rag_pipeline.Qdrant_hybrid_rag.indexing import extract_content_from_pdf
from ragas.testset.generator import TestsetGenerator

def generate_testset(query):
    # Initialize OpenAI LLM (GPT-3/4 models)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7,
                 api_key='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',
                 base_url = " https://api.groq.com/openai/v1/chat"
                 )


    # documents = [
    #     Document(page_content=point.payload.get("content", ""), metadata=point.payload.get("metadata", {}))
    #     for point in results["combined_results"]
    # ]
    # combined_context = "\n".join([doc.payload.get("content", "") for doc in results["combined_results"]])
    #
    # qa_chain = {
    #             "context":combined_context,
    #             "question": query,
    #         }
    #
    # response = llm_chain.invoke(qa_chain)
    # print("evaluation",response)
    # output = {
    #     "query": query,  # The input query
    #     "result": response,  # The LLM-generated answer
    #     "source_documents": documents  # Retrieved documents
    # }
    # print(output)
    #
    #
    # llm = initialize_llm()
    # langchain_llm = LangchainLLMWrapper(llm)
    # llm_chain = create_llm_chain(llm)
    #
    # results = hybrid_search(query)
    dense_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    langchain_embeddings = LangchainEmbeddingsWrapper(dense_embedding_model)

    generator = TestsetGenerator.from_langchain(generator_llm=llm, embeddings=langchain_embeddings, critic_llm=llm)

    chunks = extract_content_from_pdf("data/raw/LSTM.pdf")
    pages = [ Document(page_content=doc.page_content, metadata=doc.metadata) for doc in chunks[:5] ]

    # Change resulting question type distribution
    distributions = {
        simple: 0.3,
        multi_context: 0.2,
        reasoning: 0.5
    }

    testset = generator.generate_with_langchain_docs(pages, 10, distributions)
    print(testset.to_pandas())