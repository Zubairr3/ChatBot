import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def dummy_gpu_pass():
    pass

def generate_response(user_message, history):
    if history is None:
        history = []
        
    if not user_message or not str(user_message).strip():
        return "", history
    
    bot_reply = bot.get_response(str(user_message))
    
    history.append({"role": "user", "content": str(user_message)})
    history.append({"role": "assistant", "content": str(bot_reply)})
    
    return "", history

# -----------------------------------------------------
# Classic Monochrome "Human-Made" CSS Theme
# -----------------------------------------------------
custom_css = """
body, .gradio-container {
    background-color: #050505 !important;
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    color: #E5E7EB !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

.header-panel {
    background: #0F0F0F !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    padding: 16px 24px !important;
    margin-bottom: 12px !important;
}

.header-title {
    color: #FFFFFF !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    letter-spacing: 0.5px;
}

.gr-accordion {
    background-color: #0F0F0F !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
}

.chatbot-area {
    background: #0A0A0A !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    height: 400px !important; /* Fixed height forces input box up */
    max-height: 45vh !important;
    margin-bottom: 12px !important;
}

.message.user {
    background: #262626 !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 12px 16px !important;
}
.message.bot {
    background: #000000 !important;
    color: #D1D5DB !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    line-height: 1.5 !important;
}

.input-box {
    background: #0F0F0F !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
}
.input-box textarea {
    color: #FFFFFF !important;
    font-size: 1rem !important;
    padding: 12px !important;
}
.input-box textarea:focus {
    border-color: #52525B !important;
    box-shadow: none !important;
}

.send-btn {
    background: #171717 !important;
    border: 1px solid #333333 !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    height: 100% !important;
}
.send-btn:hover {
    background: #262626 !important;
}

.footer-text {
    text-align: center;
    color: #737373;
    font-size: 0.85rem;
    margin-top: 15px;
}
"""

theme = gr.themes.Monochrome(
    primary_hue="neutral",
    neutral_hue="neutral"
).set(
    block_background_fill="transparent",
    block_border_color="transparent"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Assistant") as demo:
    
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.4rem;">🏥</span>
                <h1 class='header-title'>Hospital Review Assistant</h1>
            </div>
        """)
        
        with gr.Accordion("✨ What is this bot trained on? (Click to expand)", open=False):
            gr.Markdown("""
            This AI assistant is trained entirely on **real hospital patient reviews and feedback data**. 
            
            Instead of reading through hundreds of individual comments manually, you can simply ask this bot a question or provide a keyword, and it will instantly summarize answers regarding:
            * ⏱️ **Wait Times:** How long patients wait for appointments or tests.
            * 🩺 **Medical Care:** The quality of treatment and staff attentiveness.
            * 🧼 **Cleanliness:** Room comfort and overall hospital hygiene.
            * 💳 **Billing & Admin:** Administrative clarity and support.
            """)

    chat_history = gr.Chatbot(
        type="messages", 
        elem_classes="chatbot-area", 
        show_label=False,
        avatar_images=(None, None) 
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type a keyword (e.g. 'wait') or question here...",
            show_label=False,
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Send", elem_classes="send-btn", scale=2)

    gr.Examples(
        examples=[
            "wait",
            "cleanliness",
            "What do patients say about wait times for tests?",
            "How do patients rate the medical care and staff behavior?"
        ],
        inputs=user_input
    )

    # THE UPDATED PORTFOLIO FOOTER
    gr.HTML("""
        <div class="footer-text">
            AI Portfolio Project • Python | LangChain |AI Model Integration | Gradio
        </div>
    """)

    send_btn.click(
        fn=generate_response, 
        inputs=[user_input, chat_history], 
        outputs=[user_input, chat_history]
    )
    user_input.submit(
        fn=generate_response, 
        inputs=[user_input, chat_history], 
        outputs=[user_input, chat_history]
    )

if __name__ == "__main__":
    demo.launch()