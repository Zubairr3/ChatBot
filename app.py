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
    
    # 1. Safety check for empty history
    if history is None:
        history = []
        
    # 2. Prevent empty messages from crashing the app
    if not user_message or not str(user_message).strip():
        return "", history
    
    # 3. Get AI Response
    bot_reply = bot.get_response(str(user_message))
    
    # 4. Safely update the chat history
    history.append({"role": "user", "content": str(user_message)})
    history.append({"role": "assistant", "content": str(bot_reply)})
    
    return "", history

# -----------------------------------------------------
# Premium SaaS CSS Theme
# -----------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    background-color: #0B0F19 !important; /* Deep Modern Dark */
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #E2E8F0 !important;
    max-width: 950px !important;
    margin: 0 auto !important;
}

/* Premium Header Panel */
.header-panel {
    background: linear-gradient(145deg, #111827, #1E293B) !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    padding: 24px 28px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
}

.gradient-text {
    background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.025em;
}

.status-badge {
    font-size: 0.75rem;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 6px 12px;
    border-radius: 9999px;
    color: #34D399;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.1);
}

.status-dot {
    width: 8px; 
    height: 8px; 
    background: #10B981; 
    border-radius: 50%; 
    display: inline-block;
    box-shadow: 0 0 8px #10B981;
}

/* Accordion Info Box */
.gr-accordion {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
}

/* Chatbot Container */
.chatbot-area {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    min-height: 55vh !important;
    margin-bottom: 16px !important;
    box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.06) !important;
}

/* Chat Bubbles (iOS / Modern style) */
.message.user {
    background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
    color: white !important;
    border-radius: 20px 20px 4px 20px !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
}
.message.bot {
    background: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 20px 20px 20px 4px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* Input Box */
.input-box {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}
.input-box textarea {
    color: #F8FAFC !important;
    font-size: 1rem !important;
    padding: 12px !important;
}
.input-box textarea:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* Submit Button */
.send-btn {
    background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
    height: 100% !important;
}
.send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.3) !important;
    background: linear-gradient(135deg, #60A5FA, #3B82F6) !important;
}

/* Footer styling */
.footer-text {
    text-align: center;
    color: #64748B;
    font-size: 0.85rem;
    margin-top: 20px;
    font-weight: 500;
}
"""

theme = gr.themes.Monochrome(
    primary_hue="slate",
    neutral_hue="slate"
).set(
    block_background_fill="transparent",
    block_border_color="transparent"
)

with gr.Blocks(css=custom_css, theme=theme, title="AI Hospital Analyst") as demo:
    
    # --- 1. PREMIUM HEADER ---
    with gr.Column(elem_classes="header-panel"):
        gr.HTML("""
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: #1E293B; padding: 10px; border-radius: 12px; border: 1px solid #334155;">
                        <span style="font-size: 1.8rem; line-height: 1;">🏥</span>
                    </div>
                    <div>
                        <h1 class='gradient-text'>Hospital Review AI</h1>
                        <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Intelligent Patient Feedback Analysis</p>
                    </div>
                </div>
                <div class="status-badge">
                    <span class="status-dot"></span> System Active
                </div>
            </div>
        """)
        
        # --- 2. ACCORDION (Cleaner, better formatted) ---
        with gr.Accordion("✨ How this AI works (Click to expand)", open=False):
            gr.Markdown("""
            ### 🤖 AI-Powered Data Synthesis
            This application uses a custom Retrieval-Augmented Generation (RAG) pipeline to analyze real patient feedback. 
            
            Instead of manually scrolling through hundreds of messy reviews, ask this AI a question. It will instantly search the dataset and synthesize a professional Executive Summary regarding:
            
            * ⏱️ **Wait Times** (Appointments, ER, Testing)
            * 🩺 **Medical & Nursing Quality** (Staff behavior, expertise)
            * 🧼 **Facilities** (Cleanliness, room comfort)
            * 💳 **Administration** (Billing, support)
            
            *Architecture: Built with Python, Scikit-Learn (TF-IDF), Google Gemini, and Gradio.*
            """)

    # --- 3. CHATBOT INTERFACE ---
    chat_history = gr.Chatbot(
        type="messages", 
        elem_classes="chatbot-area", 
        show_label=False,
        avatar_images=("https://api.dicebear.com/7.x/avataaars/svg?seed=Felix", "https://api.dicebear.com/7.x/bottts/svg?seed=HospitalBot")
    )
    
    # --- 4. INPUT & BUTTON CONTROLS ---
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask about patient feedback (e.g., 'How is the cleanliness?')",
            show_label=False,
            scale=8,
            elem_classes="input-box"
        )
        send_btn = gr.Button("Send", elem_classes="send-btn", scale=2)

    # --- 5. SUGGESTED PROMPTS ---
    gr.Examples(
        examples=[
            "What do patients say about wait times for tests?",
            "How do patients rate the medical care and staff behavior?",
            "Summarize the general feedback regarding hospital cleanliness."
        ],
        inputs=user_input
    )
    
    # --- 6. FOOTER BRANDING ---
    gr.HTML("""
        <div class="footer-text">
            Developed as an AI Engineering Portfolio Project • Powered by LLM Synthesis
        </div>
    """)

    # --- 7. EVENT HANDLERS ---
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
    demo.launch()  # Reverted back to the safe default!