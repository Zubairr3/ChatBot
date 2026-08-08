import gradio as gr
import spaces
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def chat_interface(message, history):
    return bot.get_response(message)

# -----------------------------------------------------
# Enterprise Obsidian & Indigo CSS
# -----------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    background-color: #09090b !important; /* Pitch dark, clean background */
    font-family: 'Inter', sans-serif !important;
    color: #f4f4f5 !important;
}

/* Sleek, Compact Enterprise Header */
.brand-header {
    background: linear-gradient(90deg, #18181b 0%, #27272a 100%) !important;
    padding: 16px 24px !important;
    border-radius: 12px !important;
    margin-bottom: 16px !important;
    border: 1px solid #3f3f46 !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.brand-title {
    color: #f4f4f5 !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-subtitle {
    color: #a1a1aa !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
}

/* Clean Accordion Info Box */
.accordion-box {
    background-color: #18181b !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
}

.info-content {
    color: #d4d4d8 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

.info-content strong {
    color: #818cf8 !important; /* Subtle indigo highlight */
}

/* Chat Area */
.chatbot-area {
    min-height: 65vh !important; /* Responsive height */
    border: 1px solid #3f3f46 !important;
    border-radius: 12px !important;
    background-color: #09090b !important;
}

/* User Message Bubble */
.message.user {
    background-color: #3730a3 !important; /* Professional Deep Indigo */
    color: #ffffff !important;
    border: none !important;
}
.message.user p { color: #ffffff !important; }

/* Bot Message Bubble */
.message.bot {
    background-color: #18181b !important;
    color: #f4f4f5 !important;
    border: 1px solid #3f3f46 !important;
}
.message.bot p { color: #f4f4f5 !important; }

/* FIX: Clearly visible Input Box */
.gradio-textbox {
    background-color: #18181b !important;
    border: 1px solid #52525b !important; /* Lighter border so it stands out */
    border-radius: 8px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}
.gradio-textbox textarea {
    color: #f4f4f5 !important;
}
"""

theme = gr.themes.Default(
    primary_hue="indigo",
    neutral_hue="zinc"
).set(
    body_background_fill="#09090b",
    block_background_fill="#18181b",
    block_border_color="#3f3f46",
    button_primary_background_fill="#3730a3",
    button_primary_text_color="#ffffff"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Bot") as demo:
    # 1. Compact Navbar Header
    gr.HTML("""
        <div class="brand-header">
            <div class="brand-title">🏥 Hospital Review Bot</div>
            <div class="brand-subtitle">AI-Powered Patient Feedback Intelligence</div>
        </div>
    """)
    
    # 2. Accordion Box
    with gr.Accordion("✨ Click here to know about this ChatBot", open=False, elem_classes="accordion-box"):
        gr.HTML("""
        <div class="info-content">
            <p style="margin-bottom: 8px;"><strong>What is this bot trained on?</strong></p>
            <p style="margin-bottom: 8px;">This AI system is trained on thousands of real patient feedback records. It instantly analyzes and answers your questions regarding:</p>
            <ul style="padding-left: 20px; margin-top: 0;">
                <li><strong>Wait Times:</strong> Diagnostics, ER responsiveness, and appointments.</li>
                <li><strong>Medical Care:</strong> Doctor expertise, nursing staff, and overall treatment quality.</li>
                <li><strong>Facilities:</strong> Cleanliness, room comfort, and billing clarity.</li>
            </ul>
        </div>
        """)

    # 3. Chat Interface with BUG FIX applied
    gr.ChatInterface(
        fn=chat_interface,
        type="messages",
        chatbot=gr.Chatbot(
            height="65vh",
            elem_classes="chatbot-area",
            type="messages" # <-- THIS FIXES THE CRASH
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