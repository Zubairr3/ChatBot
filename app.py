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
# Muted Sage & Warm Stone Executive CSS
# -----------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

body, .gradio-container {
    background-color: #1c1917 !important; /* Warm Dark Stone Background */
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #f5f5f4 !important;
}

/* Executive Header Panel */
.header-panel {
    background: #292524 !important;
    border: 1px solid #44403c !important;
    border-radius: 14px !important;
    padding: 22px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

.header-title {
    color: #e7e5e4 !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    margin: 0 0 4px 0 !important;
}

/* Chatbot Area */
.chatbot-area {
    background: #1c1917 !important;
    border: 1px solid #44403c !important;
    border-radius: 14px !important;
    min-height: 60vh !important;
    margin-bottom: 12px !important;
}

/* Message Bubbles - Muted Sage & Neutral Stone */
.message.user {
    background: #4a6b5d !important; /* Muted Sage Green */
    color: #ffffff !important;
    border: none !important;
}
.message.bot {
    background: #292524 !important;
    color: #e7e5e4 !important;
    border: 1px solid #44403c !important;
}

/* Explicit Send Button */
.send-btn {
    background: #588157 !important; /* Elegant Muted Sage */
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    transition: background 0.2s ease !important;
    cursor: pointer !important;
}
.send-btn:hover {
    background: #3a5a40 !important;
}

/* Clean Textbox */
.input-box {
    border-radius: 10px !important;
    border: 1px solid #57534e !important;
    background: #292524 !important;
}
.input-box textarea {
    color: #f5f5f4 !important;
    font-size: 1rem !important;
}
"""

theme = gr.themes.Monochrome(
    primary_hue="stone",
    neutral_hue="stone"
).set(
    block_background_fill="transparent",
    block_border_color="transparent"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    # 1. Custom Header
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <h1 class='header-title'>🏥 Hospital Review Bot</h1>
            <p style='color:#a8a29e; margin:0; font-size:1rem;'>Patient Feedback Intelligence System</p>
        """)
        
        with gr.Accordion("✨ Click here to know about this ChatBot", open=False):
            gr.Markdown("""
            **What is this bot trained on?**  
            This AI system indexes real patient feedback records to provide structured summaries on:
            *   **Wait Times:** Diagnostics, ER responsiveness, and appointments.
            *   **Medical Care:** Doctor expertise, nursing staff, and treatment quality.
            *   **Facilities:** Cleanliness, room comfort, and billing clarity.
            """)

    # 2. The Chat Display
    chat_history = gr.Chatbot(type="messages", elem_classes="chatbot-area", show_label=False)
    
    # 3. Manual Input Row with Explicit Button
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type any questions here in space to ask...",
            show_label=False,
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Send ➔", elem_classes="send-btn", scale=2)

    # 4. Interactive Examples
    gr.Examples(
        examples=[
            "What do patients say about wait times for tests?",
            "How do patients rate the medical care?",
            "Summarize general feedback regarding hospital cleanliness."
        ],
        inputs=user_input
    )

    # 5. Wire up the logic
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