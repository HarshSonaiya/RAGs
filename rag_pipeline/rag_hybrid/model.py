from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

def initialize_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

def load_quantized_model(model_name: str):
    # Example function to load a quantized model (implement as needed)
    quantization_config = BitsAndBytesConfig()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config
    )
    return model

def create_pipeline(model_name: str, tokenizer):
    model = load_quantized_model(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)
