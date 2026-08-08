import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def chat_interface(message, history):
    return bot.get_response(message)

# Premium Dark-Mode Glassmorphism CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

body, .gradio-container {
    background: linear-gradient(135deg, #020617 0%, #1e1b4b 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #f8fafc !important;
}

/* Frosted Glass Header */
.brand-header {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 35px;
    border-radius: 24px;
    margin-bottom: 30px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    text-align: center;
}

.brand-title {
    color: #38bdf8 !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    margin: 0 0 12px 0 !important;
    letter-spacing: -1px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
}

.brand-subtitle {
    color: #94a3b8 !important;
    font-size: 1.1rem !important;
    margin: 0 !important;
    font-weight: 400;
}

/* Frosted Glass Info Box */
.info-box {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-left: 6px solid #818cf8 !important;
    padding: 30px !important;
    border-radius: 20px !important;
    margin-bottom: 30px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
}

.info-box h4 {
    color: #818cf8 !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    margin: 0 0 16px 0 !important;
}

.info-box p {
    color: #cbd5e1 !important;
    font-size: 1.1rem !important;
    line-height: 1.7 !important;
}

.info-box ul {
    color: #cbd5e1 !important;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
}

.info-box strong {
    color: #38bdf8 !important;
    font-weight: 600 !important;
}
"""

# Apply native dark theme settings
theme = gr.themes.Monochrome(
    primary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="transparent",
    block_background_fill="rgba(15, 23, 42, 0.4)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_radius="20px",
    button_primary_background_fill="#38bdf8",
    button_primary_text_color="#020617",
    button_secondary_background_fill="rgba(255, 255, 255, 0.05)",
    button_secondary_text_color="#f8fafc"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    gr.HTML("""
        <div class="brand-header">
            <h1 class="brand-title">
                🏥 Hospital Review Bot
            </h1>
            <p class="brand-subtitle">
                Advanced AI Synthesis & RAG Analysis
            </p>
        </div>
        
        <div class="info-box">
            <h4>✨ What is this AI trained on?</h4>
            <p style="margin-bottom: 12px;">This system analyzes thousands of real patient feedback records instantly:</p>
            <ul>
                <li><strong>Wait Times:</strong> Diagnostics, ER responsiveness, and appointments.</li>
                <li><strong>Medical Care:</strong> Doctor expertise, nursing staff, and overall treatment quality.</li>
                <li><strong>Facilities:</strong> Cleanliness, room comfort, and billing clarity.</li>
            </ul>
        </div>
    """)

    gr.ChatInterface(
        fn=chat_interface,
        type="messages",
        examples=[
            "What do patients say about wait times for tests?",
            "How do patients rate the medical care?",
            "Summarize general feedback regarding hospital cleanliness."
        ],
    )

if __name__ == "__main__":
    demo.launch()