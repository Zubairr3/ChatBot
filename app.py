import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def chat_interface(message, history):
    return bot.get_response(message)

# Unique "Sunrise Coral & Warm Ivory" CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

body, .gradio-container {
    background-color: #fafaf9 !important; /* Warm ivory background */
    font-family: 'Poppins', sans-serif !important;
    color: #292524 !important;
}

/* Vibrant Gradient Header */
.brand-header {
    background: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%) !important;
    padding: 28px !important;
    border-radius: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 15px 35px rgba(244, 63, 94, 0.25) !important;
    text-align: center !important;
    border: none !important;
}

.brand-title {
    color: #ffffff !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.5px !important;
}

.brand-subtitle {
    color: #ffe4e6 !important;
    font-size: 1.1rem !important;
    margin: 0 !important;
    font-weight: 400 !important;
}

/* Accordion Text Styling */
.info-content {
    color: #44403c !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
}

.info-content strong {
    color: #f43f5e !important;
}

/* Chatbot visibility enforcement */
.chatbot-container {
    background: #ffffff !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05) !important;
    border: 1px solid #e7e5e4 !important;
}
"""

# Apply native light theme settings with Rose/Coral accents
theme = gr.themes.Soft(
    primary_hue="rose",
    neutral_hue="stone",
).set(
    body_background_fill="#fafaf9",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e7e5e4",
    block_radius="16px",
    button_primary_background_fill="#f43f5e",
    button_primary_text_color="#ffffff"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    # 1. Vibrant Header
    gr.HTML("""
        <div class="brand-header">
            <h1 class="brand-title">🏥 Hospital Review Bot</h1>
            <p class="brand-subtitle">AI-Powered Patient Feedback Intelligence</p>
        </div>
    """)
    
    # 2. Clean, Collapsible Info Box
    with gr.Accordion("✨ Click here to see what information is available", open=False):
        gr.HTML("""
        <div class="info-content">
            <p style="margin-bottom: 10px;"><strong>This AI system analyzes real patient feedback records instantly to answer your questions on:</strong></p>
            <ul style="padding-left: 20px; margin-top: 0;">
                <li><strong>Wait Times:</strong> Diagnostics, ER responsiveness, and appointments.</li>
                <li><strong>Medical Care:</strong> Doctor expertise, nursing staff, and overall treatment quality.</li>
                <li><strong>Facilities:</strong> Cleanliness, room comfort, and billing clarity.</li>
            </ul>
        </div>
        """)

    # 3. Chat Interface with Custom Placeholder
    gr.ChatInterface(
        fn=chat_interface,
        type="messages",
        textbox=gr.Textbox(
            placeholder="Type any questions here in space to ask...", 
            container=False, 
            scale=7
        ),
        examples=[
            "What do patients say about wait times for tests?",
            "How do patients rate the medical care?",
            "Summarize general feedback regarding hospital cleanliness."
        ],
    )

if __name__ == "__main__":
    demo.launch()