from datasets import Dataset
from rag_pipeline.Qdrant_hybrid_rag.retriever import hybrid_search
from rag_pipeline.Qdrant_hybrid_rag.generator import initialize_llm,create_llm_chain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings, LangchainEmbeddingsWrapper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Function to compute cosine similarity between two vectors
def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Function to compute Euclidean distance between two vectors
def compute_euclidean_distance(vec1, vec2):
    return euclidean_distances([vec1], [vec2])[0][0]

# Function to compute BLEU score between two sentences (ground truth and generated)
def compute_bleu_score(reference, hypothesis):
    smoothing_function = SmoothingFunction()
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing_function)


def main() :
    questions = [
        "What is the main advantage of weight guessing in neural networks?",
        "How does weight guessing compare to other algorithms for realistic tasks?",
        "Why does weight guessing become infeasible for certain tasks?",
        "What capability do Schmidhuber’s hierarchical chunker systems have?",
        "How does noise affect the performance of chunker systems?",
        "What advantage does LSTM have over chunker systems in terms of noise?",
        "What is the purpose of the architecture designed for constant error flow?",
        "How does the multiplicative input gate unit protect memory contents?",
        "What is the function of the multiplicative output gate unit?",
        "How does the new architecture improve upon the naive approach?",
        "What problems are solved by the gradient-based optimization methods?",
        "Explain the concept of local predictability in chunker systems.",
        "What are the key components of the LSTM network?",
        "Describe the training process of hierarchical chunker systems.",
        "How do the memory cells in LSTM networks affect learning long-term dependencies?"
    ]

    ground_truth = [
        "The main advantage of weight guessing is that it avoids long-time-lag problems by randomly initializing network weights until the network correctly classifies all training sequences.",
        "Weight guessing solves many problems faster than algorithms proposed by Bengio et al. (1994) and others but is not ideal for tasks requiring many parameters or high precision.",
        "Weight guessing becomes infeasible for tasks needing high precision or a large number of parameters, as it does not scale well.",
        "Schmidhuber’s hierarchical chunker systems can bridge arbitrary time lags if there is local predictability across the subsequences.",
        "The performance of chunker systems deteriorates with increasing noise levels and less compressible input sequences.",
        "LSTM handles increasing noise and less compressible sequences better than chunker systems by using gating mechanisms to regulate memory.",
        "The purpose of the architecture is to allow constant error flow through self-connected units while avoiding the disadvantages of the naive approach.",
        "The multiplicative input gate unit protects memory contents from irrelevant inputs by controlling how much new information is added to the memory cell.",
        "The multiplicative output gate unit protects other units from irrelevant memory contents by regulating how much of the memory contents affect the network output.",
        "The new architecture improves by allowing constant error flow and preventing irrelevant inputs and memory contents from disrupting the network's functioning.",
        "Gradient-based optimization methods adjust the network weights to minimize loss functions, thus solving problems related to inefficient training.",
        "Local predictability in chunker systems means that if subsequences have predictable patterns, the system can effectively bridge time lags.",
        "Key components of the LSTM network include memory cells, input gates, output gates, and forget gates, which help manage long-term dependencies.",
        "Hierarchical chunker systems are trained using sequences with local predictability to handle time lags, but their performance is affected by noise and sequence compressibility.",
        "Memory cells in LSTM networks help learning long-term dependencies by storing information over long periods and allowing gradients to flow through these cells without vanishing."
    ]
    # answers  = []
    # contexts = []

    llm = initialize_llm()
    llm_chain = create_llm_chain(llm)

    generated_answers = []  # Store generated answers from your LLM chain
    retrieved_contexts = []  # Store contexts from Qdrant hybrid search
    qdrant_embeddings = []  # Store the embeddings for retrieved contexts
    generated_embeddings = []  # Store embeddings for generated answers

    # traversing each question and passing into the chain to get answer from the system
    for question in questions:
        results = hybrid_search(question)
        combined_context = "\n".join([doc.payload.get("content", "") for doc in results["combined_results"]])
        answers.append(llm_chain.invoke({"question": question, "context": combined_context})['text'])
        contexts.append([combined_context])
    #
    # data = {
    #     "question": questions,
    #     "answer": answers,
    #     "contexts": contexts,
    #     "ground_truth": ground_truth
    # }
    # dataset = Dataset.from_dict(data)
    #
    # faithfulness.llm = llm
    # answer_relevancy.llm = llm
    # context_recall.llm = llm
    # context_precision.llm = llm
    #
    # result = evaluate(
    #     dataset=dataset,
    #     metrics=[
    #         context_precision,
    #         context_recall,
    #         faithfulness,
    #         answer_relevancy,
    #     ]
    # )
    #
    # df = result.to_pandas()
    # print(df)

if __name__ == "__main__":
    main()