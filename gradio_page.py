import gradio as gr
from rag_service import OpenAIRAG
from dotenv import load_dotenv
import os # 1. Import the 'os' module

load_dotenv()

# Initialize RAG
# This part assumes your 'rag_index' folder is present in the deployment
rag = OpenAIRAG()
rag.load_index("rag_index")
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

demo = gr.ChatInterface(
    fn=rag_chat,
    type="messages",
    autofocus=False,
    title="Telenursing Chatbot",
    description="Tanyakan apa saja tentang telenursing. Kami siap membantu Anda!",
    theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray"),
    examples=[        # Provide example questions for users
        "Apa itu telenursing?",
        "Bagaimana cara kerja telenursing?",
        "Apa manfaat telenursing bagi pasien?"
    ],
)

# 2. Add the deployment logic here
if __name__ == "__main__":
    # Get the port from the environment variable Railway provides, default to 7860
    port = int(os.environ.get("PORT", 7860))
    
    # Launch the app to be accessible within the container and on the assigned port
    demo.launch(server_name="0.0.0.0", server_port=port)