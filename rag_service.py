import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class OpenAIRAG:
    """
    A class to create and interact with a Retrieval-Augmented Generation (RAG)
    system using OpenAI's models and FAISS for vector storage.
    """
    def __init__(self, llm_model_name="gpt-3.5-turbo"):
        """
        Initializes the RAG system by setting up the API key and loading the
        OpenAI language and embedding models.

        Args:
            llm_model_name (str): The name of the OpenAI model to use for generation.
        """
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("Error: OPENAI_API_KEY environment variable not set.")

        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        
        self.db = None
        self.retriever = None
        self.qa_chain = None
        print("OpenAIRAG initialized successfully.")

    def create_index(self, document_path: str, index_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Loads a document, splits it into chunks, creates embeddings, and builds a
        FAISS vector store (index), then saves it to a local path.

        Args:
            document_path (str): The file path of the document to index.
            index_path (str): The local folder path to save the FAISS index.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
        """
        print(f"Loading document from: {document_path}")
        loader = TextLoader(document_path, encoding="utf-8")
        documents = loader.load()

        print("Splitting document into chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        print(f"Split document into {len(docs)} chunks.")

        print("Creating FAISS vector store...")
        self.db = FAISS.from_documents(docs, self.embeddings)
        
        print(f"Saving index to local path: {index_path}")
        self.db.save_local(index_path)
        print("Index created and saved successfully.")

    def load_index(self, index_path: str):
        """
        Loads a pre-existing FAISS index from a local path.

        Args:
            index_path (str): The folder path of the saved FAISS index.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path not found: {index_path}")
            
        print(f"Loading index from: {index_path}")
        # When loading, we must provide the same embedding function used to create the index
        self.db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        print("Index loaded successfully.")

    def setup_rag_chain(self):
        """
        Sets up the retriever and the RetrievalQA chain with a custom prompt.
        """
        if self.db is None:
            raise ValueError("Vector store (index) is not loaded. Please run create_index() or load_index() first.")
            
        print("Setting up RAG chain...")
        self.retriever = self.db.as_retriever()

        # Define the prompt template
        template = """
        Anda adalah Telenursing SDIDTK. Tujuan Anda adalah bertindak sebagai jembatan digital antara orang tua, anak, dan tenaga kesehatan untuk memastikan perkembangan anak yang optimal sejak dini dengan menggabungkan pemantauan, deteksi, edukasi, dan intervensi.
        Gunakan potongan konteks berikut untuk menjawab pertanyaan pengguna. Jika Anda tidak tahu jawabannya, cukup katakan bahwa Anda tidak tahu, jangan mencoba membuat jawaban.

        Konteks: {context}

        Pertanyaan: {question}

        Jawaban yang Membantu:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Inject the custom prompt
        )
        print("RAG chain is ready.")

    def query(self, question: str) -> dict:
        """
        Asks a question to the RAG chain. The retriever will find chunks relevant
        to this question.
        """
        if not self.qa_chain:
            raise RuntimeError("RAG chain is not set up. Please run setup_rag_chain() first.")
            
        print(f"Processing query: '{question}'")
        # Pass ONLY the user's question to the chain
        result = self.qa_chain.invoke({"query": question})
        return result