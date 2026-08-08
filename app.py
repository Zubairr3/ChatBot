import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

# 1. Initialize the backend engine
bot = HospitalReviewBot()

# 2. Decorate the routing function so ZeroGPU hardware boots without crashing
@spaces.GPU
def chat_interface(message, history):
    # message is a string; history is handled cleanly by type="messages"
    return bot.get_response(message)

# 3. Build the modern UI
demo = gr.ChatInterface(
    fn=chat_interface,
    type="messages",
    title="🏥 Hospital Review Bot",
    description="A Gemini-powered Retrieval-Augmented Generation (RAG) system to analyze and summarize hospital feedback.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()