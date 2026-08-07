"""Gradio app entrypoint for Hugging Face Spaces.
This file wraps the Rag_Char.ask/get_status functions into a simple Gradio Chat UI.
"""

try:
    import gradio as gr
except Exception:
    raise SystemExit("Gradio is required. Please install dependencies from requirements.txt")

from Rag_Char import ask, get_status


def respond(user_question, chat_history):
    answer = ask(user_question or "")
    chat_history = chat_history or []
    chat_history.append((user_question, answer))
    return chat_history, ""


with gr.Blocks(title="Hospital Review Bot") as demo:
    gr.Markdown("# 🏥 Hospital Review Bot\nAsk a question about patient reviews and get answers based on available context.")
    chatbot = gr.Chatbot(label="Conversation")
    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your question here...", label="Your question")
        submit = gr.Button("Send")
    status_html = "<small>Mode: full (API key set)</small>" if get_status() == "full" else "<small>Mode: fallback (API key not set)</small>"
    gr.HTML(status_html)

    submit.click(respond, inputs=[user_input, chatbot], outputs=[chatbot, user_input])


if __name__ == "__main__":
    # In Spaces the app will be launched automatically; this helps for local testing
    demo.launch()
