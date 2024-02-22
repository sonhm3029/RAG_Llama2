from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Any

import time


model_file = "models/llama-2-13b-chat.Q5_K_M.gguf"
callbacks = [StreamingStdOutCallbackHandler()]


def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.9,
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


# if __name__ == "__main__":
#     while True:
#         question = input("User: ")
#         template = """<|im_start|>system
#         Dùng thông tin sau đây để trả lời người dùng:
#         Em đang phân vân có nên lựa chọn ngành Công nghệ Kỹ thuật Điện tử Viễn thông hay không vì sợ ra trường khó xin việc. Thầy, Cô có thể tư vấn giúp em được không ạ?
#         Trả lời:
#         Chào em!
#         Hiện nay tổng số nhân lực của nhóm ngành Công nghệ kỹ thuật Điện tử Viễn thông, công nghệ thông tin là khoảng 556 ngàn người và dự báo đến năm 2020 sẽ cần lượng nhân lực khoảng 758 ngàn người. Theo trung tâm dự báo nhu cầu nhân lực và thông tin thị trường TP Hồ Chí Minh, nhu cầu nhân sự nhóm ngành nghề này trong giai đoạn 2020-2025 vẫn có nhu cầu tuyển dụng rất lớn, có thể lên đến 16.200 người/năm. Trong nhiều năm tới nhu cầu nhân lực ngành này còn cao hơn rất nhiều khi các tập đoàn sản xuất thiết bị điện tử hàng đầu thế giới tới Việt Nam. Nên học ngành này em không cần lo lắng về vấn đề khó xin việc.
#         Chúc em may mắn!
#         <|im_end|>
#         <|im_start|>user
#         {question}<|im_end|>
#         <|im_start|>assistant
#         """

#         prompt = create_prompt(template)
#         llm_chain = create_simple_chain(prompt, llm)
#         print("iViVi: ")
#         response = llm_chain.stream({"question": question})
#         for res in response:
#             print("end")
if __name__ == "__main__":
    while True:
        question = input("User: ")
        template = """[INST] <<SYS>>
        You are a very accurate assistant. Always response/answer in Vietnamese, don't return other language.
        <</SYS>>
        {question}[/INST]
        """

        prompt = create_prompt(template)
        llm_chain = create_simple_chain(prompt, llm)
        print("iViVi: ")
        response = llm_chain.stream({"question": question})
        for res in response:
            print("end")
