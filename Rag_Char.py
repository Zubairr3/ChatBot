import os
import re
import pandas as pd
import logging
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HospitalReviewBot:
    def __init__(self, data_path="reviews.csv"):
        self.data_path = data_path
        self.retriever = None
        self.llm = None
        self.prompt = None
        self.setup_rag()

    def setup_rag(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY not found. Defaulting to local fallback.")
            return
            
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset '{self.data_path}' is missing.")
            return

        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            
            loader = DataFrameLoader(df, page_content_column=text_col)
            documents = loader.load()

            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vectorstore = FAISS.from_documents(documents, embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

            self.prompt = ChatPromptTemplate.from_template(
                "Answer the following question based only on the provided context:\n\n<context>\n{context}\n</context>\n\nQuestion: {question}"
            )
            logger.info("Gemini RAG pipeline initialized successfully.")
            
        except Exception as e:
            logger.error(f"Initialization Error: {e}")
            self.retriever = None

    def get_response(self, user_query):
        query_clean = user_query.strip().lower()

        # 1. Human Chitchat & Intent Interception
        if re.fullmatch(r"(hi|hello|hey|hey there|hi there|good morning|good afternoon|good evening|greetings)", query_clean):
            return (
                "👋 **Hello! Welcome to the Hospital Review Assistant.**\n\n"
                "I am an AI bot trained on patient reviews and feedback datasets. "
                "I can analyze feedback regarding **wait times, medical care quality, staff behavior, cleanliness, and billing**.\n\n"
                "💡 **Try asking me:**\n"
                "* *'What are the main complaints regarding test wait times?'*\n"
                "* *'How do patients rate the medical and nursing care?'*\n"
                "* *'Summarize overall feedback on emergency department services.'*"
            )

        if any(w in query_clean for w in ["thanks", "thank you", "thx", "appreciate it"]):
            return "😊 You're very welcome! Feel free to ask if you need any more insights about hospital patient reviews."

        if any(w in query_clean for w in ["bye", "goodbye", "see you", "cya"]):
            return "👋 Goodbye! Have a great day and stay healthy!"

        if any(w in query_clean for w in ["who are you", "what can you do", "what is this", "help", "about"]):
            return (
                "🏥 **About Hospital Review Bot:**\n\n"
                "I am a Retrieval-Augmented Generation (RAG) assistant that queries patient review logs to help you quickly assess hospital performance.\n\n"
                "**Trained Data Categories:**\n"
                "1. ⏱️ **Wait Times**: Diagnostics, ER, and consultations.\n"
                "2. 🩺 **Care Quality**: Doctor expertise, nurse responsiveness, treatment satisfaction.\n"
                "3. 🧼 **Facilities**: Hospital cleanliness, room comfort, and billing clarity."
            )

        # 2. Main RAG Pipeline Execution
        if not self.retriever or not self.llm:
            return self._local_fallback(user_query)
            
        try:
            source_docs = self.retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": user_query})
            
            answer = response.content if hasattr(response, "content") else str(response)
            
            if source_docs:
                answer += "\n\n### 📑 **Source Citations:**\n"
                for idx, doc in enumerate(source_docs):
                    snippet = doc.page_content[:120].replace("\n", " ").strip()
                    answer += f"* **Review {idx+1}:** \"{snippet}...\"\n"
                    
            return answer
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return "**System Error:** Unable to process the request at this time."

    def _local_fallback(self, query):
        if not os.path.exists(self.data_path):
            return "**Notice:** Local data unavailable."
            
        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            keywords = query.lower().split()
            matches = df[df[text_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keywords))]
            
            if matches.empty:
                return "ℹ️ **No exact matches found.** Try asking about wait times, doctors, nursing, or cleanliness."
                
            answer = "⚠️ **Fallback Mode (API Key Missing):** Displaying top matching review record:\n\n"
            answer += f"**Patient Record:** \"{matches.iloc[0][text_col]}\"\n"
            return answer
            
        except Exception as e:
            return "**Notice:** Could not complete local retrieval."