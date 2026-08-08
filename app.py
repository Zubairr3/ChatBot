import gradio as gr
from Rag_Char import HospitalReviewBot

# 1. Initialize the backend engine
bot = HospitalReviewBot()

# 2. Define the routing function
def chat_interface(message, history):
    return bot.get_response(message)

# 3. Build the modern UI
demo = gr.ChatInterface(
    fn=chat_interface,
    title="🏥 Hospital Review Bot",
    description="A Gemini-powered Retrieval-Augmented Generation (RAG) system to analyze and summarize hospital feedback.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()