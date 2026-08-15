import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def dummy_gpu_pass():
    pass

def update_user_message(user_message, history):
    """Instantly clears the input box and shows the user's message in the chat."""
    if history is None:
        history = []
    if not user_message or not str(user_message).strip():
        return "", history
    
    history.append({"role": "user", "content": str(user_message)})
    return "", history

def generate_bot_response(history):
    """Calls the AI model in the background and appends the response."""
    if not history or history[-1]["role"] != "user":
        return history
    
    user_message = history[-1]["content"]
    bot_reply = bot.get_response(user_message)
    
    history.append({"role": "assistant", "content": bot_reply})
    return history

# -----------------------------------------------------
# Premium "Human-Designed" Charcoal CSS Theme
# -----------------------------------------------------
custom_css = """
/* Base Background - Very dark, but not pure black for less eye strain */
body, .gradio-container {
    background-color: #0d0d0d !important; 
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #f3f4f6 !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Header & Panels - Slightly raised slate color */
.header-panel {
    background: #171717 !important;
    border: 1px solid #262626 !important;
    border-radius: 10px !important;
    padding: 20px 24px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5) !important;
}

.header-title {
    color: #ffffff !important;
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    letter-spacing: -0.01em !important;
}

.gr-accordion {
    background-color: #171717 !important;
    border: 1px solid #262626 !important;
    border-radius: 8px !important;
    color: #d1d5db !important;
}

/* Chat Area */
.chatbot-area {
    background: #121212 !important; 
    border: 1px solid #262626 !important;
    border-radius: 10px !important;
    height: 420px !important; /* Fixed height prevents scrolling */
    max-height: 50vh !important;
    margin-bottom: 16px !important;
    box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.2) !important;
    padding: 12px !important;
}

/* Chat Bubbles - Humanized asymmetrical shaping */
.message.user {
    background: #27272a !important; /* Distinct dark zinc color */
    color: #ffffff !important;
    border-radius: 14px 14px 4px 14px !important; /* iMessage style curve */
    border: 1px solid #3f3f46 !important;
    padding: 12px 16px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.message.bot {
    background: #171717 !important; 
    color: #e5e5e5 !important;
    border-radius: 14px 14px 14px 4px !important; /* iMessage style curve */
    border: 1px solid #262626 !important;
    padding: 14px 18px !important;
    line-height: 1.6 !important; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
}

/* Forcing solid bullet points for readability */
.message.bot ul {
    list-style-type: disc !important;
    padding-left: 20px !important;
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}
.message.bot li {
    margin-bottom: 6px !important;
}

/* Input box styling */
.input-box {
    background: #171717 !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease !important;
}
.input-box:focus-within {
    border-color: #71717a !important; 
}
.input-box textarea {
    color: #ffffff !important;
    font-size: 1rem !important;
    padding: 14px !important;
    resize: none !important;
}

/* Tactile Send Button */
.send-btn {
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease, transform 0.1s ease !important;
    height: 100% !important;
}
.send-btn:hover {
    background: #3f3f46 !important;
}
.send-btn:active {
    transform: scale(0.96) !important; /* Physical push-down effect */
}

/* Sleek Scrollbars */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #3f3f46;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #52525b;
}

.footer-text {
    text-align: center;
    color: #71717a;
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

    gr.HTML("""
        <div class="footer-text">
            AI Portfolio Project • Python | LangChain | Generative LLMs | Gradio
        </div>
    """)

    send_btn.click(
        fn=update_user_message, 
        inputs=[user_input, chat_history], 
        outputs=[user_input, chat_history],
        queue=False 
    ).then(
        fn=generate_bot_response, 
        inputs=[chat_history], 
        outputs=[chat_history]
    )
    
    user_input.submit(
        fn=update_user_message, 
        inputs=[user_input, chat_history], 
        outputs=[user_input, chat_history],
        queue=False 
    ).then(
        fn=generate_bot_response, 
        inputs=[chat_history], 
        outputs=[chat_history]
    )

if __name__ == "__main__":
    demo.launch()