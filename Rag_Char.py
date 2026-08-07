import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Callable

# Load environment files but avoid using secrets at import-time for safety
load_dotenv(".env")
load_dotenv("ke.env", override=True)

BASE_DIR = Path(__file__).resolve().parent
REVIEWS_CSV_PATH = BASE_DIR / "reviews.csv"
REVIEWS_FAISS_PATH = BASE_DIR / "faiss_index"

logger = logging.getLogger("rag_char")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Lazy global state
_state = {
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    "vector_db": None,
    "retriever": None,
    "chat_model": None,
}

def _get_embedding_function():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning("Embedding init failed — running in fallback mode: %s", e)
        return None

def _get_vector_db():
    if _state.get("vector_db") is not None:
        return _state["vector_db"]

    embedding_function = _get_embedding_function()
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.document_loaders.csv_loader import CSVLoader
    except Exception as e:
        logger.warning("Vectorstore imports failed: %s", e)
        return None

    if not REVIEWS_FAISS_PATH.exists():
        if not REVIEWS_CSV_PATH.exists():
            logger.error("CSV not found at %s", REVIEWS_CSV_PATH)
            return None
        logger.info("Creating FAISS index...")
        loader = CSVLoader(file_path=str(REVIEWS_CSV_PATH), source_column="review")
        documents = loader.load()
        if embedding_function:
            db = FAISS.from_documents(documents, embedding_function)
            db.save_local(str(REVIEWS_FAISS_PATH))
            _state["vector_db"] = db
            return db
        else:
            return None
    else:
        if embedding_function:
            logger.info("Loading existing FAISS index...")
            try:
                db = FAISS.load_local(str(REVIEWS_FAISS_PATH), embedding_function, allow_dangerous_deserialization=True)
                _state["vector_db"] = db
                return db
            except Exception as e:
                logger.warning("Failed to load FAISS index: %s", e)
                return None
    return None

def _init_chat_model() -> Callable[[str], str]:
    """Return a callable that accepts a prompt string and returns a string response.
    This function avoids raising on import-time and uses a safe fallback when model
    initialization fails or no API key is present.
    """
    if _state.get("chat_model") is not None:
        return _state["chat_model"]

    key = _state.get("google_api_key")
    if key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=key)

            def model_callable(prompt_text: str) -> str:
                # Try multiple invocation styles defensively
                try:
                    if hasattr(model, "generate"):
                        # Some model wrappers expose generate()
                        return model.generate(prompt_text)
                    if hasattr(model, "run"):
                        return model.run(prompt_text)
                    # If model is callable
                    if callable(model):
                        return model(prompt_text)
                    return str(model)
                except Exception as e:
                    logger.warning("Model call failed: %s", e)
                    return "[Model error]"

            _state["chat_model"] = model_callable
            return model_callable
        except Exception as e:
            logger.warning("Google Generative AI init failed: %s", e)

    # Fallback model
    def dummy_chat_model(prompt_text: str) -> str:
        return (
            "[Fallback] No API/model available.\n"
            "Prompt delivered to model (truncated):\n" + prompt_text[:1000]
        )

    _state["chat_model"] = dummy_chat_model
    return dummy_chat_model

def init_retriever():
    if _state.get("retriever") is not None:
        return _state["retriever"]

    db = _get_vector_db()
    if db is None:
        logger.info("Vector DB not available; retriever will be None.")
        return None

    try:
        _state["retriever"] = db.as_retriever(search_kwargs={"k": 10})
        return _state["retriever"]
    except Exception as e:
        logger.warning("Failed to create retriever: %s", e)
        return None

def _format_docs(docs):
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

def ask(question: str) -> str:
    """Main public function to ask a question. Returns a string answer.
    This function is safe to call without any API keys (it will use fallbacks).
    """
    chat_model = _init_chat_model()
    retriever = init_retriever()

    if retriever is not None:
        try:
            # Try a simple retrieval + prompt approach
            docs = retriever.get_relevant_documents(question)
            context = _format_docs(docs)
            prompt_text = f"Context:\n{context}\n\nQuestion: {question}"
            try:
                return chat_model(prompt_text)
            except Exception as e:
                logger.warning("Chat model invocation failed: %s", e)
        except Exception as e:
            logger.warning("Retrieval/inference error: %s", e)

    # Fallback behavior: return an excerpt from CSV
    try:
        with open(REVIEWS_CSV_PATH, encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()[1:]
        top = lines[0].split(",")[2] if lines else "No reviews available."
    except Exception:
        top = "No reviews available (file missing or unreadable)."
    return f"[Fallback answer] {top[:300]}"

# Small helper used by the Gradio UI and tests
def get_status() -> str:
    return "full" if _state.get("google_api_key") else "fallback"

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", type=str)
    parser.add_argument("--ui", action="store_true")
    args = parser.parse_args()

    if args.ui:
        try:
            import gradio as gr
        except Exception:
            logger.error("Gradio is not installed — please pip install -r requirements.txt")
            return

        def gradio_respond(user_question, chat_history):
            answer = ask(user_question)
            chat_history = chat_history or []
            chat_history.append((user_question, answer))
            return chat_history, ""

        with gr.Blocks(title="Hospital Review Bot") as demo:
            gr.Markdown("# 🏥 Hospital Review Bot\nAsk a question about patient reviews and get answers based on available context.")
            chatbot = gr.Chatbot(label="Conversation")
            with gr.Row():
                user_input = gr.Textbox(placeholder="Type your question here...", label="Your question")
                submit = gr.Button("Send")
            status = gr.HTML("<small>API key set: running full mode.</small>" if _state.get("google_api_key") else "<small>API key not set: running fallback mode.</small>")

            submit.click(gradio_respond, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

        demo.launch()
    elif args.question:
        print(ask(args.question))
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()
