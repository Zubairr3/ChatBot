import sys
import os

# Patch huggingface_hub to add missing HfFolder for Gradio compatibility
try:
    from huggingface_hub import _login
except ImportError:
    pass

# Monkey-patch HfFolder if it doesn't exist
import huggingface_hub
if not hasattr(huggingface_hub, 'HfFolder'):
    class HfFolder:
        path_token = os.path.expanduser("~/.huggingface")
        
        @classmethod
        def save_token(cls, token):
            pass
        
        @classmethod
        def get_token(cls):
            return os.environ.get("HF_TOKEN", None)
        
        @classmethod
        def delete_token(cls):
            pass
    
    huggingface_hub.HfFolder = HfFolder

import gradio as gr
from Rag_Char import HospitalReviewBot

# Initialize Backend
bot = HospitalReviewBot(data_path="reviews.csv")

def chat_interface(user_message, history):
    if not user_message.strip():
        return "", history
    response = bot.get_response(user_message)
    history.append((user_message, response))
    return "", history

custom_css = """
.hero-banner {
    background: linear-gradient(135deg, #4f46e5 0%, #0d9488 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.hero-banner h1 {
    margin: 0;
    font-size: 2.2em;
    font-weight: 700;
}
.hero-banner p {
    margin: 8px 0 0 0;
    font-size: 1.1em;
    opacity: 0.95;
}
"""

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal")

with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.HTML('''
    <div class="hero-banner">
        <h1>🏥 Hospital Review Bot</h1>
        <p>Intelligent Insights and Retrieval-Augmented Generation for Healthcare Feedback</p>
    </div>
    ''')
    chatbot = gr.Chatbot(height=520, show_label=False, bubble_full_width=False)
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your question about hospital reviews here...", scale=4, show_label=False)
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])
    submit_btn.click(chat_interface, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
