import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()



def update_user_message(user_message, history):
    if history is None:
        history = []
    if not user_message or not str(user_message).strip():
        return "", history
    
    history.append({"role": "user", "content": str(user_message)})
    return "", history

def generate_bot_response(history):
    if not history or history[-1]["role"] != "user":
        return history
    
    user_message = history[-1]["content"]
    bot_reply = bot.get_response(user_message)
    
    history.append({"role": "assistant", "content": bot_reply})
    return history

# -----------------------------------------------------
# Modern Clinical Light Mode CSS (Ocean Teal Accents)
# -----------------------------------------------------
custom_css = """
/* Soft Off-White Background */
body, .gradio-container {
    background-color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #0f172a !important; 
    max-width: 950px !important;
    margin: 0 auto !important;
    padding: 20px 10px !important;
}

/* Pure White Cards with subtle borders */
.header-panel {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

.header-title {
    color: #0f172a !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
}

.header-subtitle {
    color: #64748b !important;
    font-size: 0.95rem !important;
    margin-top: 6px !important;
    margin-bottom: 0 !important;
}

/* Distinct Accordion with Contrast */
.gr-accordion {
    background-color: #F1F5F9 !important; /* Differentiates from the pure white background */
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #334155 !important;
}

/* Chat Canvas */
.chatbot-area {
    background: #FFFFFF !important; 
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    padding: 16px !important;
}

/* Chat Bubbles - Ocean Teal for User */
.message.user {
    background: #0284C7 !important; /* Trustworthy Ocean Teal */
    color: #ffffff !important; 
    border-radius: 14px 14px 4px 14px !important;
    border: none !important;
    padding: 12px 18px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2) !important;
}

/* Chat Bubbles - Soft Light Grey for Bot */
.message.bot {
    background: #F8FAFC !important; 
    color: #0f172a !important; 
    border-radius: 14px 14px 14px 4px !important;
    border: 1px solid #E2E8F0 !important;
    padding: 16px 20px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important; 
}

.message.bot ul {
    list-style-type: disc !important;
    padding-left: 20px !important;
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}
.message.bot li {
    margin-bottom: 6px !important;
}

/* Floating Input Box Container */
.input-box {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.input-box:focus-within {
    border-color: #0284C7 !important; 
    box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.2) !important;
}
.input-box textarea {
    color: #0f172a !important; 
    font-size: 1rem !important;
    padding: 14px !important;
    resize: none !important;
}

/* Primary Ocean Teal Send Button */
.send-btn {
    background: #0284C7 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    height: 100% !important;
    box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2) !important;
}
.send-btn:hover {
    background: #0369a1 !important; /* Darkens slightly on hover */
}
.send-btn:active {
    transform: scale(0.97) !important;
}

/* Custom Clean Scrollbars */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}

.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 20px;
}
"""

# Implementing the requested Gradio Soft Theme
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
).set(
    block_background_fill="transparent",
    block_border_color="transparent"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Assistant") as demo:
    
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <div style="display: flex; flex-direction: column; gap: 4px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 1.6rem;">🏥</span>
                    <h1 class='header-title'>Hospital Review Assistant</h1>
                </div>
                <!-- FIXED: 1-Line Info Subtitle directly under the main title -->
                <p class='header-subtitle'>Powered by patient review records covering wait times, care quality, and facility ratings.</p>
            </div>
        """)
        
        # FIXED: Better contrast on the accordion to separate it from pure white background
        with gr.Accordion("📌 What is this bot trained on? (Click to expand)", open=False):
            gr.Markdown("""
            This AI assistant is trained entirely on **real hospital patient reviews and feedback data**. 
            
            Instead of reading through hundreds of individual comments manually, you can simply ask this bot a question or provide a keyword, and it will instantly summarize answers regarding:
            * ⏱️ **Wait Times:** How long patients wait for appointments or tests.
            * 🩺 **Medical Care:** The quality of treatment and staff attentiveness.
            * 🧼 **Cleanliness:** Room comfort and overall hospital hygiene.
            * 💳 **Billing & Admin:** Administrative clarity and support.
            """)

    # FIXED: Added default greeting message and set fixed height to 450
    initial_message = [
        {"role": "assistant", "content": "👋 Hello! I can help you analyze hospital reviews. Ask me about wait times, staff quality, or search by keywords."}
    ]
    
    chat_history = gr.Chatbot(
        value=initial_message,
        type="messages", 
        elem_classes="chatbot-area", 
        show_label=False,
        avatar_images=(None, None),
        height=450  
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type a keyword (e.g. 'wait') or question here...",
            show_label=False,
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Send", elem_classes="send-btn", scale=2)

    # FIXED: Replaced generic examples with highly actionable, clickable user queries
    gr.Examples(
        examples=[
            "What do patients say about emergency room wait times?",
            "Summarize overall feedback on doctor communication.",
            "List common complaints regarding billing."
        ],
        inputs=user_input
    )

    gr.HTML("""
        <div class="footer-text">
            AI Portfolio Project • Python | LangChain | Generative LLMs | Gradio
        </div>
    """)

    # Click Handlers
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