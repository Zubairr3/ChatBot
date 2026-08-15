import gradio as gr
import spaces
import re
import random
from Rag_Char import HospitalReviewBot

bot = HospitalReviewBot()

@spaces.GPU
def dummy_gpu_pass():
    pass

# FIX 4: Prevent empty bubbles and duplicate triggers
def update_user_message(user_message, history):
    if history is None:
        history = []
    
    # If the user clicks send with an empty box, do absolutely nothing.
    if not user_message or not str(user_message).strip():
        return user_message, history
    
    # Otherwise, append the user's message
    history.append({"role": "user", "content": str(user_message).strip()})
    return "", history

def generate_bot_response(history):
    # Only generate a response if the very last message was actually from the user
    if not history or history[-1]["role"] != "user":
        return history
    
    user_message = str(history[-1]["content"])
    clean_msg = re.sub(r'[^\w\s]', '', user_message.strip().lower())
    
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "thanks", "thank you", "ok", "okay", "good", "great", "awesome", "perfect", "thankyou"]
    
    # FIX 5: Varied, dynamic responses for greetings and farewells
    if clean_msg in greetings or (len(clean_msg.split()) <= 2 and any(g in clean_msg for g in greetings)):
        bot_reply = random.choice([
            "👋 Hello! I am the Hospital Review Assistant. Ask me anything about the hospital's wait times, staff quality, or cleanliness.",
            "Hi there! How can I help you analyze the hospital reviews today?",
            "Greetings! Ready to search through the patient feedback. What would you like to know?"
        ])
    elif clean_msg in farewells or (len(clean_msg.split()) <= 3 and any(f in clean_msg for f in farewells)):
        bot_reply = random.choice([
            "You're welcome! 😊 Have a great day!",
            "Happy to help! Let me know if you need anything else.",
            "Anytime! I'm here if you have more questions.",
            "Glad I could assist! Have a wonderful day!"
        ])
    else:
        # If it's a real query, pass it to your AI model
        bot_reply = bot.get_response(user_message)
    
    history.append({"role": "assistant", "content": bot_reply})
    return history

# -----------------------------------------------------
# Safe, Non-Destructive CSS
# -----------------------------------------------------
custom_css = """
.gradio-container {
    max-width: 950px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* FIX 1: Force Title to Pure White for Dark Mode */
.header-title {
    color: #FFFFFF !important;
}
.header-subtitle {
    color: #94A3B8 !important;
}

/* Ensure text doesn't vanish while processing */
.message-wrap.generating {
    opacity: 1 !important;
}
"""

# FIX 2: Let Gradio's native theme handle the chat bubbles perfectly
theme = gr.themes.Soft(
    primary_hue="sky",
    neutral_hue="slate"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Assistant") as demo:
    
    gr.HTML("""
        <div style="padding-top: 15px; padding-bottom: 10px;">
            <h1 class="header-title" style="font-size: 1.8rem; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 10px;">
                🏥 Hospital Review Assistant
            </h1>
            <p class="header-subtitle" style="font-size: 1rem; margin-top: 5px;">
                Summarizing actual patient feedback on wait times, care quality, and facility ratings.
            </p>
        </div>
    """)
    
    with gr.Accordion("💡 How does this AI work? (Behind the scenes)", open=False):
        gr.Markdown("""
        Think of this assistant as a **smart search engine** for hospital feedback. 
        
        Instead of making a human read through thousands of comments manually, this AI does it instantly:
        1. **You ask a question** (e.g., "Are the nurses friendly?").
        2. **The AI scans** a secure database of over 1,000 real patient reviews.
        3. **It gathers the facts** and summarizes only what the patients actually said.
        
        *This ensures you get honest, fact-based answers based on real patient experiences, without the AI making anything up.*
        """)

    # FIX 3: Increased height to 500 and ensured autoscroll is natively handled
    chat_history = gr.Chatbot(
        value=[{"role": "assistant", "content": "👋 Hello! I am the Hospital Review Assistant. Ask me anything about the hospital's wait times, staff quality, or cleanliness."}],
        type="messages", 
        show_label=False,
        avatar_images=(None, None),
        height=500,
        autoscroll=True
    )
    
    gr.HTML("<p style='font-size: 0.85rem; color: #64748B; margin-bottom: 4px;'>💡 <b>Tip:</b> Click an example below, or copy a <i>Related Question</i> from the chat.</p>")
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type a keyword (e.g. 'wait') or question here...",
            show_label=False,
            container=False, 
            scale=8
        )
        send_btn = gr.Button("Send", variant="primary", scale=2)

    gr.Examples(
        examples=[
            "What do patients say about emergency room wait times?",
            "Summarize overall feedback on doctor communication.",
            "List common complaints regarding billing."
        ],
        inputs=user_input
    )

    gr.HTML("""
        <div style="text-align: center; color: #64748B; font-size: 0.85rem; margin-top: 30px; margin-bottom: 20px;">
            AI Portfolio Project • Built with Python, Gradio & Generative AI
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