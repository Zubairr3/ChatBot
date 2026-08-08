import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

# Initialize backend engine
bot = HospitalReviewBot()

@spaces.GPU
def chat_interface(message, history):
    return bot.get_response(message)

# Custom High-Contrast CSS for Medical Dashboard UI
custom_css = """
/* Global Container Styling */
.gradio-container {
    background-color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
}

/* Brand Header */
.brand-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    padding: 28px !important;
    border-radius: 16px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.15) !important;
    border: 1px solid #334155 !important;
}

.brand-title {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    margin: 0 !important;
}

.badge-icon {
    background: #0284c7 !important;
    width: 44px !important;
    height: 44px !important;
    border-radius: 12px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.5rem !important;
    box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3) !important;
}

.brand-subtitle {
    color: #cbd5e1 !important;
    font-size: 0.95rem !important;
    margin-top: 8px !important;
    font-weight: 400 !important;
}

/* Info Box - FIXED TEXT VISIBILITY & CONTRAST */
.info-box {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-left: 5px solid #0284c7 !important;
    padding: 20px 24px !important;
    border-radius: 14px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}

.info-box h4 {
    color: #0f172a !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    margin: 0 0 10px 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.info-box p {
    color: #334155 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    margin: 0 0 12px 0 !important;
    font-weight: 500 !important;
}

.info-box ul {
    color: #334155 !important;
    margin: 0 !important;
    padding-left: 22px !important;
}

.info-box li {
    color: #334155 !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    margin-bottom: 8px !important;
}

.info-box strong {
    color: #0369a1 !important;
    font-weight: 700 !important;
}

/* Chatbot Container Integration */
.chatbot-container {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}
"""

# Configure Theme
theme = gr.themes.Soft(
    primary_hue="sky",
    neutral_hue="slate",
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_radius="14px",
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    # 1. Custom Visual Header
    gr.HTML("""
        <div class="brand-header">
            <h1 class="brand-title">
                <span class="badge-icon">🩺</span> Hospital Review Bot
            </h1>
            <p class="brand-subtitle">
                Gemini-powered Retrieval-Augmented Generation (RAG) System for Medical Feedback Intelligence
            </p>
        </div>
    """)

    # 2. Onboarding Information Box (Strict Light Mode Typography)
    gr.HTML("""
        <div class="info-box">
            <h4>💡 What is this bot trained on?</h4>
            <p>
                This AI system indexes real patient feedback to provide instant, summarized insights across key categories:
            </p>
            <ul>
                <li><strong>Wait Times & Scheduling:</strong> Diagnostic test results, ER response speed, and appointment delays.</li>
                <li><strong>Care & Staff Quality:</strong> Doctor expertise, nursing responsiveness, and bedside care quality.</li>
                <li><strong>Facilities & Billing:</strong> Hospital cleanliness, room amenities, and administrative experience.</li>
            </ul>
        </div>
    """)

    # 3. Interactive Chat Interface
    gr.ChatInterface(
        fn=chat_interface,
        type="messages",
        examples=[
            "Hi",
            "What do patients say about wait times for tests?",
            "How is the quality of nursing and medical care?",
            "Summarize general feedback regarding hospital cleanliness."
        ],
    )

if __name__ == "__main__":
    demo.launch()