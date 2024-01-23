from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings




class VectorDB:
    def __init__(
        self,
        vector_db_path="vectorstores/db_faiss",
        pdf_data_path="data",
        embed_model="sentence-transformers/all-MiniLM-l6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
        self.pdf_data_path = pdf_data_path
        self.vector_db_path = vector_db_path

    # Create vector db from text
    def create_db_from_text(self, raw_text: str):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        chunks = text_splitter.split_text(raw_text)
        
        db = FAISS.from_texts(texts=chunks, embedding=self.embedding_model)
        db.save_local(self.vector_db_path)
        
        return db

    def create_db_from_files(self, ext="pdf", loader=PyPDFLoader):
        loader = DirectoryLoader(self.pdf_data_path, glob=f"*.{ext}", loader_cls = loader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        db = FAISS.from_documents(chunks, self.embedding_model)
        db.save_local(self.vector_db_path)
        
        return db
    
if __name__ == "__main__":
    vectordb = VectorDB(pdf_data_path="data/word")
    vectordb.create_db_from_files(ext="docx", loader=Docx2txtLoader)
    

