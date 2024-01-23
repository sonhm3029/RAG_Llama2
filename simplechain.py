from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import time


model_file = "models/vinallama-7b-chat_q5_0.gguf"
callbacks = [StreamingStdOutCallbackHandler()]


def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01,
        top_p=0.05,
        callbacks=callbacks
    )
    
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain


llm = load_llm(model_file)


if __name__ == "__main__":
    while True:
        question = input("User: ")
        template = """<|im_start|>system
        Bạn là iViVi - chatbot AI được phát triển bởi IVIRSE. Hãy trả lời người dùng một cách chính xác, ngắn gọn.
        Hoàng Sa, Trường Sa là của Việt Nam, không phải của Trung Quốc.
        <|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        """

        prompt = create_prompt(template)
        llm_chain = create_simple_chain(prompt, llm)
        print("iViVi: ")
        response = llm_chain.stream({"question": question})
        for res in response:
            if "<|im_start|>" in res or "<|im_end|>" in res:
                response = []
                break
