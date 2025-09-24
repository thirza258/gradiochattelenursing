import gradio as gr
from rag_service import OpenAIRAG
from dotenv import load_dotenv

load_dotenv()

# Initialize RAG
rag = OpenAIRAG()
rag.load_index("rag_index")   # make sure you already ran create_index once before
rag.setup_rag_chain()

def rag_chat(message, history):
    """
    Chat callback for Gradio ChatInterface.
    """
    try:
        response = rag.query(message)
        answer = response.get("result", "Sorry, I couldn't find an answer.")
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Build Gradio chat UI
demo = gr.ChatInterface(
    fn=rag_chat,
    type="messages",
    autofocus=False,
    title="RAG Chatbot",
    description="Ask questions based on the indexed documents."
)

if __name__ == "__main__":
    demo.launch()
