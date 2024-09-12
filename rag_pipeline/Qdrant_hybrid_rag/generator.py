from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq
from rag_pipeline.Qdrant_hybrid_rag.retriever import hybrid_search
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

def initialize_llm(model_name="llama3-8b-8192"):

    return ChatGroq(temperature=0.3,model_name=model_name, api_key=groq_api_key)


def create_llm_chain(llm):

    template = """You are an AI assistant for answering questions about the various documents from the user.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown:"""

    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    return LLMChain(llm=llm, prompt=prompt)

def answer_query(query: str) -> str:
    print("retrieval begins")
    # Retrieve documents using the hybrid search function
    results = hybrid_search(query)
    print("retrieval ends")
    # Combine results into a single context string
    combined_context = "\n".join([doc.payload.get("content", "") for doc in results["combined_results"]])

    llm = initialize_llm()
    llm_chain = create_llm_chain(llm)

    response = llm_chain.invoke({"question": query, "context":combined_context})['text']
    return response