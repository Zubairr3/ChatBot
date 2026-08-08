import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

# Initialize backend
bot = HospitalReviewBot()

@spaces.GPU
def generate_response(user_message, history):
    """Custom state management function for the manual Blocks UI"""
    if not user_message.strip():
        return "", history
    
    # Fetch AI response (will gracefully handle API limits)
    bot_reply = bot.get_response(user_message)
    
    # Append to history using the modern Gradio 5 format
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_reply})
    
    # Return empty string to clear textbox, and the updated history
    return "", history

# -----------------------------------------------------
# Deep Space & Neon Cyan Glassmorphism CSS
# -----------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #f8fafc !important;
}

/* Glassmorphism Header */
.header-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

.header-title {
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 5px 0 !important;
}

/* Chatbot Area */
.chatbot-area {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 16px !important;
    min-height: 60vh !important;
    margin-bottom: 15px !important;
}

/* Message Bubbles */
.message.user {
    background: linear-gradient(90deg, #38bdf8 0%, #0284c7 100%) !important;
    color: #ffffff !important;
    border: none !important;
}
.message.bot {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Explicit Send Button */
.send-btn {
    background: linear-gradient(90deg, #818cf8 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    cursor: pointer !important;
}
.send-btn:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 15px rgba(129, 140, 248, 0.5) !important;
}

/* Clean Textbox */
.input-box {
    border-radius: 12px !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    background: rgba(15, 23, 42, 0.8) !important;
}
.input-box textarea {
    color: #f8fafc !important;
    font-size: 1.05rem !important;
}
"""

theme = gr.themes.Monochrome(
    primary_hue="sky",
    neutral_hue="slate"
).set(
    block_background_fill="transparent",
    block_border_color="transparent"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    # 1. Custom Header
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <h1 class='header-title'>🏥 Hospital Review Bot</h1>
            <p style='color:#94a3b8; margin:0; font-size:1.1rem;'>Advanced Patient Feedback Intelligence</p>
        """)
        
        with gr.Accordion("✨ Tap to see what this AI is trained on", open=False):
            gr.Markdown("""
            *   **Wait Times:** Diagnostics, ER responsiveness, and appointments.
            *   **Medical Care:** Doctor expertise, nursing staff, and overall treatment quality.
            *   **Facilities:** Cleanliness, room comfort, and billing clarity.
            """)

    # 2. The Chat Display
    chat_history = gr.Chatbot(type="messages", elem_classes="chatbot-area", show_label=False)
    
    # 3. Manual Input Row with Explicit Button
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your question here and press Send...",
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

    # 5. Wire up the logic (Triggers on Button Click OR pressing Enter)
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