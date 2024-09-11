from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq

groq_api_key = <your-api_key>


def create_llm_chain(model_name="llama3-8b-8192"):
    template = """You are an AI assistant for answering questions about the various documents from the user.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown:"""

    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    llm = ChatGroq(temperature=0, model_name=model_name, api_key=groq_api_key)
    return LLMChain(llm=llm, prompt=prompt)


async def answer_query(llm_chain, retriever, query: str) -> str:
    response = llm_chain.invoke({"question": query, "chat_history": []})["answer"]
    return response