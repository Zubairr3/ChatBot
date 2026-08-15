import gradio as gr
import spaces
import re
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
    
    user_message = str(history[-1]["content"])
    
    # 1. FIXED: Robust Greeting & Farewell Logic
    # Strips punctuation to perfectly catch words like "thanks!" or "bye."
    clean_msg = re.sub(r'[^\w\s]', '', user_message.strip().lower())
    
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "thanks", "thank you", "ok", "okay", "good", "great", "awesome", "perfect", "thankyou"]
    
    # Checks if the user's message is just a short greeting or farewell
    if clean_msg in greetings or (len(clean_msg.split()) <= 2 and any(g in clean_msg for g in greetings)):
        bot_reply = "👋 Hello! I am the Hospital Review Assistant. Ask me anything about the hospital's wait times, staff quality, or cleanliness."
    elif clean_msg in farewells or (len(clean_msg.split()) <= 3 and any(f in clean_msg for f in farewells)):
        bot_reply = "You're welcome! 😊 If you need to analyze any more patient feedback, I'm right here. Have a great day!"
    else:
        # If it is a real question, pass it to the AI
        bot_reply = bot.get_response(user_message)
    
    history.append({"role": "assistant", "content": bot_reply})
    return history

# -----------------------------------------------------
# 2. FIXED: Minimal, Mobile-Friendly CSS
# -----------------------------------------------------
custom_css = """
/* Keeps everything centered and perfectly sized on phones, tablets, and PCs */
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Stops the screen from flashing white or hiding text while the AI is thinking */
.generating, .translucent {
    opacity: 1 !important;
}

/* User Chat Bubble (Ocean Teal) */
.message.user {
    background-color: #0284C7 !important;
    border: none !important;
    border-radius: 14px 14px 4px 14px !important;
}
.message.user * {
    color: #FFFFFF !important; /* Forces user text to be perfectly visible */
}

/* Hides the weird white button squares on the user's message */
.message.user button {
    display: none !important;
}

/* Bot Chat Bubble (Soft Grey) */
.message.bot {
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px 14px 14px 4px !important;
}
.message.bot * {
    color: #0F172A !important;
}

/* Clean Accordion */
.gr-accordion {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
"""

# Uses Gradio's native theme builder so it doesn't break across different devices
theme = gr.themes.Soft(
    primary_hue="sky",
    neutral_hue="slate"
).set(
    body_background_fill="#F8FAFC",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E8F0",
    button_primary_background_fill="#0284C7",
    button_primary_background_fill_hover="#0369A1",
    button_primary_text_color="#FFFFFF"
)

with gr.Blocks(css=custom_css, theme=theme, title="Hospital Review Assistant") as demo:
    
    gr.HTML("""
        <div style="padding-top: 15px; padding-bottom: 10px;">
            <h1 style="font-size: 1.8rem; font-weight: 700; color: #0F172A; margin: 0; display: flex; align-items: center; gap: 10px;">
                🏥 Hospital Review Assistant
            </h1>
            <p style="color: #64748B; font-size: 1rem; margin-top: 5px;">
                Summarizing actual patient feedback on wait times, care quality, and facility ratings.
            </p>
        </div>
    """)
    
    # 3. FIXED: Plain English explanation for non-IT people (HR, recruiters, ordinary users)
    with gr.Accordion("💡 How does this AI work? (Behind the scenes)", open=False):
        gr.Markdown("""
        Think of this assistant as a **smart search engine** for hospital feedback. 
        
        Instead of making a human read through thousands of comments manually, this AI does it instantly:
        1. **You ask a question** (e.g., "Are the nurses friendly?").
        2. **The AI scans** a secure database of over 1,000 real patient reviews.
        3. **It gathers the facts** and summarizes only what the patients actually said.
        
        *This ensures you get honest, fact-based answers based on real patient experiences, without the AI making anything up.*
        """)

    chat_history = gr.Chatbot(
        value=[{"role": "assistant", "content": "👋 Hello! I can help you analyze hospital reviews. Ask me about wait times, staff quality, or search by keywords."}],
        type="messages", 
        show_label=False,
        avatar_images=(None, None),
        height=450  
    )
    
    gr.HTML("<p style='font-size: 0.85rem; color: #64748B; margin-bottom: 4px;'>💡 <b>Tip:</b> Click an example below, or copy a <i>Related Question</i> from the chat.</p>")
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type a keyword (e.g. 'wait') or question here...",
            show_label=False,
            container=False, 
            scale=8
        )
        # Uses standard Gradio "primary" variant for perfect button alignment on all devices
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
        <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; margin-top: 30px; margin-bottom: 20px;">
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