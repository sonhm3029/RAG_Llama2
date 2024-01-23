from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import time


model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"
callbacks = [StreamingStdOutCallbackHandler()]

def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01,
        callbacks=callbacks
    )
    
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = db.as_retriever(search_kwargs = {"k": 3}),
        return_source_documents = True,
        chain_type_kwargs={"prompt": prompt}
    )
    return llm_chain

def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    db = FAISS.load_local(vector_db_path, embeddings=embedding_model)
    return db

if __name__ == "__main__":
    
    db = read_vectors_db()
    start = time.time()
    llm = load_llm(model_file)
    print(f"Load LLM model in {time.time() - start} seconds")

    template = """<|im_start|>system
    Hãy sử dụng các thông tin sau đây để trả lời người dùng một cách chính xác
    {context}
    <|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """

    prompt = create_prompt(template)
    llm_chain = create_qa_chain(prompt, llm, db)
    
    start = time.time()
    print("Start thinking ...")
    question = "Chào Thầy Cô, em đang phân vân lựa chọn giữa ngành Công nghệ kỹ thuật Điện tử viễn thông với ngành Công nghệ kỹ thuật Điều khiển và Tự động hóa. Xin cho em biết sự khác nhau giữa hai ngành để có lựa chọn đúng đắn nhất?"
    # response = llm_chain.invoke({"query": question})
    # print(response)
    response = llm_chain.stream({'query': question})
    for res in response:
        print(res)
    
    
    print(f"\nReceive answer after {start - time.time()}s")

