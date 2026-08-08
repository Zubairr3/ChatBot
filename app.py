import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def generate_response(user_message, history):
    """Custom state management function for the manual Blocks UI"""
    if not user_message.strip():
        return "", history
    
    bot_reply = bot.get_response(user_message)
    
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_reply})
    
    return "", history

# -----------------------------------------------------
# Clean Developer Tool / GitHub Dark Theme CSS
# -----------------------------------------------------
custom_css = """
body, .gradio-container {
    background-color: #0d1117 !important; /* GitHub Dark Base */
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
    color: #c9d1d9 !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Developer Header Bar */
.header-panel {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
}

.header-title {
    color: #f0f6fc !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* Accordion Info Box */
.gr-accordion {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* Chatbot Container */
.chatbot-area {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    min-height: 55vh !important;
    margin-bottom: 12px !important;
}

/* Message Bubbles */
.message.user {
    background: #1f6feb !important; /* GitHub Signature Blue */
    color: #ffffff !important;
    border-radius: 6px !important;
    border: none !important;
}
.message.bot {
    background: #21262d !important; /* GitHub Surface Dark */
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* Action Button */
.send-btn {
    background: #238636 !important; /* GitHub Signature Green */
    border: 1px solid rgba(240,246,252,0.1) !important;
    color: white !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    border-radius: 6px !important;
    cursor: pointer !important;
}
.send-btn:hover {
    background: #2ea043 !important;
}

/* Input Field */
.input-box {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
.input-box textarea {
    color: #c9d1d9 !important;
    font-size: 0.95rem !important;
}
"""

theme = gr.themes.Monochrome(
    primary_hue="slate",
    neutral_hue="slate"
).set(
    block_background_fill="transparent",
    block_border_color="transparent"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    # 1. Clean Minimalist Navbar Header
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1 class='header-title'>🏥 hospital-review-bot <span style="color: #8b949e; font-weight: 400; font-size: 0.9rem;">/ main</span></h1>
                <span style="font-size: 0.85rem; background: #21262d; border: 1px solid #30363d; padding: 3px 8px; border-radius: 12px; color: #8b949e;">v1.0.0</span>
            </div>
        """)
        
        with gr.Accordion("ℹ️ System Architecture & Dataset Context", open=False):
            gr.Markdown("""
            **Trained Dataset Schema & RAG Parameters:**
            *   **Data Source:** Indexed patient review logs (`reviews.csv`).
            *   **Embeddings Model:** Google Gemini (`gemini-embedding-001`) via FAISS Vector Store.
            *   **Evaluated Domains:** Wait times, medical/nursing care quality, facility hygiene, and billing transparency.
            """)

    # 2. Chat Feed
    chat_history = gr.Chatbot(type="messages", elem_classes="chatbot-area", show_label=False)
    
    # 3. Input Controls
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask a query regarding patient feedback...",
            show_label=False,
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Submit", elem_classes="send-btn", scale=2)

    # 4. Quick Prompts
    gr.Examples(
        examples=[
            "What do patients say about wait times for tests?",
            "How do patients rate the medical care?",
            "Summarize general feedback regarding hospital cleanliness."
        ],
        inputs=user_input
    )

    # 5. Handlers
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