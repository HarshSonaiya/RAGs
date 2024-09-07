import streamlit as st
import requests
import logging
from logging_config import get_logger

# Initialize logger
logger = get_logger("streamlit")

# Title of the app
st.title("RAG Pipeline PDF Processing and Comparison")

# File uploader for PDF
uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")

# Optional: User can also input a specific query to perform
user_query = st.text_area("Optional: Enter a query (leave blank to use full PDF content)", height=100)

# Submit button
if st.button("Submit"):
    if uploaded_pdf is None:
        st.warning("Please upload a PDF document.")
        logger.warning("No PDF was uploaded by the user.")
    else:
        # Code to send the request to the backend
        with st.spinner("Processing your request..."):
            # Send the file and query to the backend
            files = {"file": (uploaded_pdf.name, uploaded_pdf, "application/pdf")}
            data = {"query": user_query} if user_query.strip() != "" else {}

            try:
                # Send request to backend
                response = requests.post(
                    "http://localhost:9000/api/ragas",  # Example endpoint
                    files=files,
                    data=data
                )
                logger.info("Request sent to FastAPI backend.")

                # Process the response
                if response.status_code == 200:

                    results = response.json()
                    logger.info("Successful response received from FastAPI backend.")

                    # Extract different responses
                    result_dense = results.get("dense_response", "No dense retriever response")
                    result_hybrid = results.get("hybrid_response", "No hybrid retriever response")
                    result_sparse = results.get("sparse_response", "No Sparse retriever response")
                    result_ragas = results.get("ragas_comparison", "No comparison data")

                    # Display the results
                    st.subheader("Response from Dense Retriever")
                    st.write(result_dense)

                    st.subheader("Response from Hybrid Retriever")
                    st.write(result_hybrid)

                    st.subheader("Response from Sparse Retriever")
                    st.write(result_sparse)

                    st.subheader("RAGAs Comparison")
                    st.write(result_ragas)

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    logger.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception(f"An error occurred: {e}")

