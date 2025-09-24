from rag_service import OpenAIRAG
from dotenv import load_dotenv
import os
import sys

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

os.makedirs("rag_index", exist_ok=True)

if __name__ == "__main__":
    rag = OpenAIRAG()
    rag.create_index("data/Telenursing.txt", "rag_index")
