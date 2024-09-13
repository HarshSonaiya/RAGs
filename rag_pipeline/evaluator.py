from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from wandb.integration.langchain.wandb_tracer import langchain

from rag_pipeline.Qdrant_hybrid_rag.retriever import hybrid_search
from rag_pipeline.Qdrant_hybrid_rag.generator import initialize_llm,create_llm_chain
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_recall,
#     context_precision,
# )
# from datasets import load_dataset, Dataset
from rag_pipeline.Qdrant_hybrid_rag.indexing import extract_content_from_pdf
#
# def generate_testset(query):
#
#     llm = initialize_llm()
#     llm_chain = create_llm_chain(llm)
#     results = hybrid_search(query)
#
#     documents = [
#         Document(page_content=point.payload.get("content", ""), metadata=point.payload.get("metadata", {}))
#         for point in results["combined_results"]
#     ]
#     combined_context = "\n".join([doc.payload.get("content", "") for doc in results["combined_results"]])
#
#     qa_chain = {
#                 "context":combined_context,
#                 "question": query,
#             }
#
#     response = llm_chain.invoke(qa_chain)
#     print("evaluation",response)
#     output = {
#         "query": query,  # The input query
#         "result": response,  # The LLM-generated answer
#         "source_documents": documents  # Retrieved documents
#     }
#     print(output)
#     # dense_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     # generator = TestsetGenerator.from_langchain(generator_llm=llm, embeddings=dense_embedding_model, critic_llm=llm)
#     #
#     # chunks = extract_content_from_pdf("data/raw/LSTM.pdf")
#     # pages = [ Document(page_content=doc.page_content, metadata=doc.metadata) for doc in chunks[:5] ]
#     #
#     # # Change resulting question type distribution
#     # distributions = {
#     #     simple: 0.3,
#     #     multi_context: 0.2,
#     #     reasoning: 0.5
#     # }
#     #
#     # testset = generator.generate_with_langchain_docs(pages, 10, distributions)
#     # print(testset.to_pandas())

from ragas.testset.generator import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain.llms import HuggingFaceHub
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

# langchain_llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B", model_kwargs={"temperature": 0.7})
langchain_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Sentence Transformer

# langchain_llm = LangchainLLMWrapper(langchain_llm)
langchain_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

llm = BaseLanguageModel(model="EleutherAI/gpt-neo-2.7B") # any langchain LLM instance


# llm_chain = create_llm_chain(llm)
# results = hybrid_search(query)

# Initialize the test set generator using your own LLM
generator = TestsetGenerator.from_langchain(generator_llm=llm, critic_llm=llm, embeddings= langchain_embeddings)
chunks = extract_content_from_pdf("data/raw/LSTM.pdf")
pages = [ Document(page_content=doc.page_content, metadata=doc.metadata) for doc in chunks[:5] ]

# Generate the test set with custom question distribution
testset = generator.generate_with_langchain_docs(
    pages,          # The    list of documents
    num_questions=10,   # Number of questions to generate
    distributions={     # Define the question types
        simple: 0.3,
        reasoning: 0.5,
        multi_context: 0.2
    }
)