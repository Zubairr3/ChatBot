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
    background-color: #050505 !important; /* Deepest black background */
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    color: #E5E7EB !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Minimalist Header Panel */
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

/* Accordion Box */
.gr-accordion {
    background-color: #0F0F0F !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
}

/* Chatbot Container - HEIGHT REDUCED TO FIX SCROLLING */
.chatbot-area {
    background: #0A0A0A !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    height: 400px !important; /* Fixed height forces input box up */
    max-height: 45vh !important;
    margin-bottom: 12px !important;
}

/* Classic Chat Bubbles */
.message.user {
    background: #262626 !important; /* Lighter charcoal for user */
    color: #FFFFFF !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 12px 16px !important;
}
.message.bot {
    background: #000000 !important; /* Pure black for bot */
    color: #D1D5DB !important; /* Soft white text for visibility */
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    line-height: 1.5 !important;
}

/* Input Box */
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

/* Submit Button */
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

with gr.Blocks(css=custom_css, theme=theme, title="Patient Feedback Analysis") as demo:
    
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.4rem;">🏥</span>
                <h1 class='header-title'>Patient Feedback Analyst</h1>
            </div>
        """)
        
        with gr.Accordion("System Instructions (Expand)", open=False):
            gr.Markdown("""
            Enter a keyword (e.g., **"wait"**, **"clean"**) or a full question to instantly search and summarize patient feedback records.
            """)

    chat_history = gr.Chatbot(
        type="messages", 
        elem_classes="chatbot-area", 
        show_label=False,
        avatar_images=(None, None) # Removed avatars for a cleaner, classic look
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type a keyword or question here...",
            show_label=False,
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Send", elem_classes="send-btn", scale=2)

    gr.HTML("""
        <div class="footer-text">
            Data Analysis Project • Built with Python & Scikit-Learn
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