import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def dummy_gpu_pass():
    pass

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
# Modern Clinical Light Mode CSS (Forced Unified Theme)
# -----------------------------------------------------
custom_css = """
/* FIX 1: Override Dark Mode Variables to Prevent Theme Fragmentation */
:root, .dark, body, .gradio-container {
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #F8FAFC !important;
    --body-background-fill: #F8FAFC !important;
    --body-text-color: #0F172A !important;
    --color-text-primary: #0F172A !important;
    --color-text-secondary: #475569 !important;
    --border-color-primary: #E2E8F0 !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: #E2E8F0 !important;
    --input-background-fill: #FFFFFF !important;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Container Sizing */
.gradio-container {
    max-width: 950px !important;
    margin: 0 auto !important;
    padding: 20px 10px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Pure White Header Cards */
.header-panel {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

.header-title {
    color: #0F172A !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
}

.header-subtitle {
    color: #64748B !important;
    font-size: 0.95rem !important;
    margin-top: 6px !important;
    margin-bottom: 0 !important;
}

/* Distinct Accordion */
.gr-accordion {
    background-color: #F1F5F9 !important; 
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #334155 !important;
}

/* FIX 2: Unified Chat Canvas (Removes the Pitch Navy) */
.chatbot-area, .chatbot-area > div, .chatbot-area .bubble-wrap {
    background: #FFFFFF !important; 
    border-color: transparent !important;
}
.chatbot-area {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    padding: 16px !important;
}

/* FIX 3: Bot Bubble Text Invisibility */
.message.bot {
    background: #F1F5F9 !important; 
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px 14px 14px 4px !important;
    padding: 16px 20px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important; 
}
/* Force all text inside the bot bubble to be high-contrast dark navy */
.message.bot p, .message.bot span, .message.bot li, .message.bot {
    color: #0F172A !important; 
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

/* User Bubbles - Ocean Teal */
.message.user {
    background: #0284C7 !important; 
    border: none !important;
    border-radius: 14px 14px 4px 14px !important;
    padding: 12px 18px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2) !important;
}
.message.user p, .message.user span, .message.user {
    color: #FFFFFF !important; 
}

/* FIX 4: Streamlined Input Box Container */
.input-box {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    overflow: hidden !important;
}
.input-box:focus-within {
    border-color: #0284C7 !important; 
    box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.2) !important;
}
.input-box textarea {
    background: transparent !important;
    color: #0F172A !important; 
    font-size: 1rem !important;
    padding: 14px !important;
    resize: none !important;
    border: none !important;
    box-shadow: none !important;
}

/* Primary Ocean Teal Send Button */
.send-btn {
    background: #0284C7 !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    height: 100% !important;
    box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2) !important;
}
.send-btn:hover {
    background: #0369A1 !important; 
}
.send-btn:active {
    transform: scale(0.97) !important;
}

/* FIX 5: Improve "Examples" Label Contrast */
[data-testid="block-info"], .label, .label-text, .gr-sample-label, .svelte-1b6s6s {
    color: #334155 !important; 
    font-weight: 700 !important;
    font-size: 0.9rem !important;
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
    color: #64748B;
    font-size: 0.85rem;
    margin-top: 20px;
}
"""

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
                <p class='header-subtitle'>Powered by patient review records covering wait times, care quality, and facility ratings.</p>
            </div>
        """)
        
        with gr.Accordion("📌 What is this bot trained on? (Click to expand)", open=False):
            gr.Markdown("""
            This AI assistant is trained entirely on **real hospital patient reviews and feedback data**. 
            
            Instead of reading through hundreds of individual comments manually, you can simply ask this bot a question or provide a keyword, and it will instantly summarize answers regarding:
            * ⏱️ **Wait Times:** How long patients wait for appointments or tests.
            * 🩺 **Medical Care:** The quality of treatment and staff attentiveness.
            * 🧼 **Cleanliness:** Room comfort and overall hospital hygiene.
            * 💳 **Billing & Admin:** Administrative clarity and support.
            """)

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
        # container=False applied here to remove the double-border shell
        user_input = gr.Textbox(
            placeholder="Type a keyword (e.g. 'wait') or question here...",
            show_label=False,
            container=False, 
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Send", elem_classes="send-btn", scale=2)

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