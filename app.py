import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def chat_interface(message, history):
    return bot.get_response(message)

# -----------------------------------------------------
# Obsidian Dark & Vibrant Violet Responsive CSS
# -----------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

body, .gradio-container {
    background-color: #121212 !important; /* True Dark Grey / Black Background */
    font-family: 'Inter', sans-serif !important;
    color: #e4e4e7 !important;
}

/* Elegant Violet Gradient Header */
.brand-header {
    background: linear-gradient(135deg, #8b5cf6 0%, #5b21b6 100%) !important;
    padding: 24px !important;
    border-radius: 16px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.25) !important;
    text-align: center !important;
    border: 1px solid #a78bfa !important;
}

.brand-title {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.5px !important;
}

.brand-subtitle {
    color: #ddd6fe !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    font-weight: 400 !important;
}

/* Accordion Info Box Styling */
.accordion-box {
    background-color: #1e1e2f !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 12px !important;
}

.info-content {
    color: #d4d4d8 !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}

.info-content strong {
    color: #a78bfa !important; /* Light Violet highlights */
}

/* ----------------------------------------------------- */
/* FIX: Chat Message Colors (Prevents Invisible Text)    */
/* ----------------------------------------------------- */

/* User Input Bubble (The "hi" message) */
.message.user {
    background-color: #8b5cf6 !important; /* Bright violet */
    color: #ffffff !important; 
    border: none !important;
}
.message.user p {
    color: #ffffff !important;
}

/* AI Response Bubble */
.message.bot {
    background-color: #27272a !important; /* Clear Dark Grey */
    color: #f4f4f5 !important;
    border: 1px solid #3f3f46 !important;
}
.message.bot p {
    color: #f4f4f5 !important;
}

/* Make Chatbot larger & responsive across Mobile/PC/Tablet */
.chatbot-area {
    min-height: 60vh !important; /* Takes up 60% of screen height automatically */
    border: 1px solid #3f3f46 !important;
    border-radius: 16px !important;
    background-color: #18181b !important;
}
"""

# Native Gradio Dark Theme Settings
theme = gr.themes.Default(
    primary_hue="violet",
    neutral_hue="zinc"
).set(
    body_background_fill="#121212",
    block_background_fill="#1e1e2f",
    block_border_color="#3f3f46",
    button_primary_background_fill="#8b5cf6",
    button_primary_text_color="#ffffff"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    # 1. Vibrant Violet Header
    gr.HTML("""
        <div class="brand-header">
            <h1 class="brand-title">🏥 Hospital Review Bot</h1>
            <p class="brand-subtitle">AI-Powered Patient Feedback Intelligence</p>
        </div>
    """)
    
    # 2. Updated Accordion Label and Content
    with gr.Accordion("✨ Click here to know about this ChatBot", open=False, elem_classes="accordion-box"):
        gr.HTML("""
        <div class="info-content">
            <p style="margin-bottom: 10px;"><strong>What is this bot trained on?</strong></p>
            <p style="margin-bottom: 10px;">This AI system is trained on thousands of real patient feedback records. It instantly analyzes and answers your questions regarding:</p>
            <ul style="padding-left: 20px; margin-top: 0;">
                <li><strong>Wait Times:</strong> Diagnostics, ER responsiveness, and appointments.</li>
                <li><strong>Medical Care:</strong> Doctor expertise, nursing staff, and overall treatment quality.</li>
                <li><strong>Facilities:</strong> Cleanliness, room comfort, and billing clarity.</li>
            </ul>
        </div>
        """)

    # 3. Dynamic, Enlarged Chat Interface
    gr.ChatInterface(
        fn=chat_interface,
        type="messages",
        chatbot=gr.Chatbot(
            height="60vh", # Increased initial size; flexes on smaller screens
            elem_classes="chatbot-area"
        ),
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