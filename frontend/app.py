import streamlit as st
import requests
import logging
from logging_config import get_logger

# Initialize logger
logger = get_logger("streamlit")

# Title of the app
st.title("RAG Pipeline PDF Processing and Comparison")

# File uploader for PDFs
uploaded_pdfs = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

# Optional: User can also input a specific query to perform
user_query = st.text_area("Enter a query", height=100)

# Submit button
if st.button("Submit"):
    if not uploaded_pdfs:
        st.warning("Please upload at least one PDF document.")
        logger.warning("No PDF was uploaded by the user.")
    elif user_query.strip() == "":
        st.warning("Please enter a query.")
        logger.warning("No query was entered by the user.")
    else:
        # Code to send the request to the backend
        with st.spinner("Processing your request..."):
            # Prepare the files for the request
            files = []
            for uploaded_file in uploaded_pdfs:
                files.append(
                    ('files', (uploaded_file.name, uploaded_file, 'application/pdf'))
                )
            try:
                # Send request to backend
                response = requests.post(
                    "http://localhost:9000/api/rag",
                    files=files,
                    data={"query": user_query},
                )
                logger.info("Request sent to FastAPI backend.")

                # Process the response
                if response.status_code == 200:
                    results = response.json()
                    logger.info("Successful response received from FastAPI backend.")

                    print(results)

                    result_hybrid = results.get("hybrid_response", "No hybrid retriever response")
                    hybrid_llm_eval = results.get("hybrid_llm_eval","No Response")
                    retriever_llm_eval = results.get("hybrid_retriever_eval","No Response")

                    result_hyde = results.get("hybrid_response", "No hybrid retriever response")
                    hyde_llm_eval = results.get("hybrid_llm_eval","No Response")
                    hyde_retriever_llm_eval = results.get("hybrid_retriever_eval","No Response")

                    st.subheader("Response from Hybrid Retriever")
                    st.write(result_hybrid)
                    st.subheader("Evaluation of Hybrid Retriever Response")
                    st.subheader("LLM Evaluation")
                    st.write(hybrid_llm_eval[0]["content"])

                    st.subheader("Retriever Evaluation")
                    st.write(retriever_llm_eval[0]["content"])

                    st.subheader("Response from HyDE Retriever")
                    st.write(result_hyde)
                    st.subheader("Evaluation of HyDE Retriever Response")
                    st.subheader("LLM Evaluation")
                    st.write(hyde_llm_eval[0]["content"])

                    st.subheader("Retriever Evaluation")
                    st.write(hyde_retriever_llm_eval[0]["content"])

                    st.write("Helpfulness measures how useful or beneficial the response is to the user.")
                    st.write("Correctness evaluates the accuracy and factual consistency of the response.")
                    st.write("Coherence assesses how well the response flows logically and maintains consistency in its ideas and language.")
                    st.write("Complexity indicates the level of difficulty or sophistication in the language and ideas presented in the response.")
                    st.write("Verbosity measures the length or wordiness of the response.")

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    logger.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception(f"An error occurred: {e}")
