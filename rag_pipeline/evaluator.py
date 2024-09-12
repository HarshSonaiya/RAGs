import ragas
from langchain_core.documents import Document
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from rag_pipeline.Qdrant_hybrid_rag.generator import initialize_llm
from rag_pipeline.Qdrant_hybrid_rag.indexing import dense_embedding_model, extract_content_from_pdf

# def generate_testset(file: str):
#
#     llm = initialize_llm()
#     generator = TestsetGenerator.from_langchain(generator_llm=llm, embeddings=dense_embedding_model, critic_llm=llm)
#
#     chunks = extract_content_from_pdf(file)
#     pages = [ Document(page_content=doc.page_content, metadata=doc.metadata) for doc in chunks ]
#
#     # Change resulting question type distribution
#     distributions = {
#         simple: 0.3,
#         multi_context: 0.2,
#         reasoning: 0.5
#     }
#
#     testset = generator.generate_with_langchain_docs(pages, 10, distributions)
#     print(testset.to_pandas())