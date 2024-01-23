from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time


model_file = "models/llama-2-13b-chat.Q4_K_M.gguf"

def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

template = """<|im_start|>system
Bạn là một trợ lý AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

prompt = create_prompt(template)
llm = load_llm(model_file)
llm_chain = create_simple_chain(prompt, llm)

start = time.time()
question = "Viết đoạn văn giới thiệu về việt nam"
response = llm_chain.invoke({"question": question})
print(response)
print(f"\n Receive response after {time.time() - start} seconds")