import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

# Initialize backend engine
bot = HospitalReviewBot()

@spaces.GPU
def chat_interface(message, history):
    return bot.get_response(message)

# Custom CSS for modern styling, readable contrast, and unified background
custom_css = """
body, .gradio-container {
    background-color: #f8fafc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.brand-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
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
    background: #0284c7;
    width: 42px;
    height: 42px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}

.brand-subtitle {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    margin-top: 6px !important;
}

.info-box {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0284c7;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: #334155;
}

.info-box h4 {
    margin: 0 0 8px 0;
    color: #0f172a;
    font-size: 1rem;
}

.info-box ul {
    margin: 0;
    padding-left: 20px;
    font-size: 0.9rem;
}
"""

with gr.Blocks(css=custom_css, title="Hospital Review Bot") as demo:
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

    # 2. Onboarding Information Box
    gr.HTML("""
        <div class="info-box">
            <h4>💡 What is this bot trained on?</h4>
            <p style="margin: 0 0 8px 0; font-size: 0.9rem;">
                This AI system indexes real patient feedback to provide summaries and insights on:
            </p>
            <ul>
                <li><strong>Wait Times & Scheduling:</strong> Diagnostic results, ER response, consultation delays.</li>
                <li><strong>Care & Staff Quality:</strong> Doctor expertise, nursing responsiveness, bedside care.</li>
                <li><strong>Facilities & Billing:</strong> Hygiene, room amenities, and administrative experience.</li>
            </ul>
        </div>
    """)

    # 3. Chat Interface with Examples
    gr.ChatInterface(
        fn=chat_interface,
        type="messages",
        examples=[
            "Hi",
            "What do patients say about wait times for tests?",
            "How is the quality of nursing and medical care?",
            "Summarize general feedback regarding hospital cleanliness."
        ],
        theme="soft"
    )

if __name__ == "__main__":
    demo.launch()