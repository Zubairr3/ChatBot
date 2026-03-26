import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env first, then ke.env to override if present
load_dotenv(".env")
load_dotenv("ke.env", override=True)
api_key = os.getenv("GOOGLE_API_KEY")

# --- 1. CONFIG & ENVIRONMENT ---
# Use os.getenv to avoid leaking keys in code
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set; falling back to local dummy model output.")

BASE_DIR = Path(__file__).resolve().parent
REVIEWS_CSV_PATH = BASE_DIR / "reviews.csv"
REVIEWS_FAISS_PATH = BASE_DIR / "faiss_index"

# --- 2. DATA LOAD & EMBEDDINGS ---

def get_embedding_function():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print("WARNING: Unable to initialize embeddings (offline/fallback).", e)
        return None


def get_vector_db():
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders.csv_loader import CSVLoader

    embedding_function = get_embedding_function()

    if not REVIEWS_FAISS_PATH.exists():
        if not REVIEWS_CSV_PATH.exists():
            raise FileNotFoundError(f"CSV not found at {REVIEWS_CSV_PATH}")
        
        print("Creating FAISS index...")
        loader = CSVLoader(file_path=str(REVIEWS_CSV_PATH), source_column="review")
        documents = loader.load()
        if embedding_function:
            db = FAISS.from_documents(documents, embedding_function)
            db.save_local(str(REVIEWS_FAISS_PATH))
            return db
        else:
            return None
    else:
        if embedding_function:
            print("Loading existing FAISS index...")
            return FAISS.load_local(
                str(REVIEWS_FAISS_PATH), 
                embedding_function, 
                allow_dangerous_deserialization=True
            )
        return None

vector_db = None
retriever = None

# --- 3. PROMPT & MODEL ---
# Note: Gemini 1.5 Flash is the current stable high-speed model
if GOOGLE_API_KEY:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        print("WARNING: Unable to initialize Google Generative AI model; using fallback.", e)
        from langchain_core.runnables import RunnableLambda
        chat_model = RunnableLambda(lambda prompt_text: "[Fallback model] API model init failed.")
else:
    # Dummy fallback model for local testing when no API key is provided.
    def dummy_chat_model(prompt_text: str) -> str:
        return (
            "[Fallback] No API key available. \n"
            "Prompt delivered to model:\n" + prompt_text[:1000]
        )

    from langchain_core.runnables import RunnableLambda
    chat_model = RunnableLambda(dummy_chat_model)

# --- 4. RAG CHAIN ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnableLambda
format_docs_runnable = RunnableLambda(format_docs)

review_chain = None

# --- 5. EXECUTION FUNCTIONS ---
def init_retriever():
    global vector_db, retriever, review_chain
    if retriever is None and GOOGLE_API_KEY:
        vector_db = get_vector_db()
        if vector_db is not None:
            retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    if retriever is not None:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        # Simplified ChatPromptTemplate usage
        review_prompt = ChatPromptTemplate.from_template("""
Your job is to use patient reviews to answer questions about their experience at a hospital.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's not from the context.
If you don't know an answer, say you don't know.

Context:
{context}

Question: {question}
""")

        review_chain = (
            {"context": retriever | format_docs_runnable, "question": RunnablePassthrough()}
            | review_prompt
            | chat_model
            | StrOutputParser()
        )
    else:
        review_chain = None


def ask(question: str) -> str:
    init_retriever()
    if review_chain is not None:
        return review_chain.invoke({"question": question})

    # Fallback when vector store or embeddings are unavailable.
    with open(REVIEWS_CSV_PATH, encoding='utf-8', errors='ignore') as f:
        data = f.read().splitlines()[1:]
    top_review = data[0].split(',')[2] if data else "No reviews available."
    return (
        "[Fallback answer] Unable to run full model chain. "
        f"Top review excerpt: {top_review[:300]}"
    )

def respond_to_user_question(question: str, history: list) -> str:
    return ask(question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", type=str)
    parser.add_argument("--ui", action="store_true")
    args = parser.parse_args()

    if args.ui:
        try:
            import gradio as gr

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
                status = gr.HTML("<small>API key not set: running fallback mode.</small>" if not GOOGLE_API_KEY else "<small>API key set: running full mode.</small>")

                submit.click(gradio_respond, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

            demo.launch()
        except ImportError:
            print("Please install gradio: pip install gradio")
    elif args.question:
        print(f"\nAnswer: {ask(args.question)}")
    else:
        parser.print_help()