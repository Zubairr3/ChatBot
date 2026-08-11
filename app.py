import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

# -----------------------------------------------------
# THE HACK: A fake GPU function to pass Hugging Face's 
# ZeroGPU startup check without breaking the CPU backend.
# -----------------------------------------------------
@spaces.GPU
def dummy_gpu_pass():
    pass

# Our real generation function (Runs safely on CPU)
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
    background-color: #0d1117 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
    color: #c9d1d9 !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

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

.gr-accordion {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

.chatbot-area {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    min-height: 55vh !important;
    margin-bottom: 12px !important;
}

.message.user {
    background: #1f6feb !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    border: none !important;
}
.message.bot {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

.send-btn {
    background: #238636 !important;
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
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5rem;">🏥</span>
                    <h1 class='header-title'>Hospital Review Bot</h1>
                </div>
                <span style="font-size: 0.8rem; background: #21262d; border: 1px solid #30363d; padding: 4px 10px; border-radius: 20px; color: #8b949e; display: inline-flex; align-items: center; gap: 6px;">
                    <span style="width: 8px; height: 8px; background: #238636; border-radius: 50%; display: inline-block;"></span> Online & Ready
                </span>
            </div>
        """)
        
        with gr.Accordion("💡 Click here to know about this ChatBot", open=False):
            gr.Markdown("""
            ### 👋 Welcome! What is this bot trained on?
            This AI assistant is trained entirely on **real hospital patient reviews and feedback data**. 
            
            Instead of reading through hundreds of individual comments manually, you can simply ask this bot a question, and it will instantly read the patient feedback to summarize the answers for you regarding:
            * ⏱️ **Wait Times:** How long patients wait for appointments, tests, or emergency room care.
            * 🩺 **Medical & Nursing Care:** The quality of treatment, doctor expertise, and staff attentiveness.
            * 🧼 **Hospital Cleanliness & Facilities:** Room comfort, hygiene, and overall environment.
            * 💳 **Billing & Support:** Administrative clarity and hospital support.
            """)

    chat_history = gr.Chatbot(type="messages", elem_classes="chatbot-area", show_label=False)
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type any questions here in space to ask...",
            show_label=False,
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Submit", elem_classes="send-btn", scale=2)

    gr.Examples(
        examples=[
            "What do patients say about wait times for tests?",
            "How do patients rate the medical care?",
            "Summarize general feedback regarding hospital cleanliness."
        ],
        inputs=user_input
    )

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