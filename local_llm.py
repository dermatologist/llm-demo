from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import tqdm
import torch
import os
import time

start_time = time.time()

# Load the model
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
llm = HuggingFacePipeline(pipeline=pipe)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

TEMPLATE = """
Context information is below.
 --------------------
 Katie presents today's session feeling “depressed.” She states that her mood has been “getting worse.”
 She reports that she has been struggling to get out of the house to enjoy social events that she is actually interested in,
 then becomes more frustrated with herself. She describes an increase in sleep issues,
 both difficulty falling asleep and staying asleep. Katie says that she “just wants to be better.”
 Katie continues to endorse low energy, loss of interest in activities, difficulty sleeping, and feelings of guilt,
 although she is not able to express the source of her guilt.
 --------------------
 Given the context information and not prior knowledge, answer the query.
 Query: {question}

 Answer:
"""

PROMPT = PromptTemplate.from_template(TEMPLATE)

_input = {
    "question": "What are Katie's complaints?"
}

chain = PROMPT | llm | StrOutputParser()

response = chain.invoke(_input)
print(response)