import os
import pandas as pd
import logging
import re
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
        self.fallback_reason = "Unknown Error"
        self.setup_rag()

    def setup_rag(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self.fallback_reason = "GOOGLE_API_KEY is missing from the active environment variables."
            logger.warning(self.fallback_reason)
            return
            
        if not os.path.exists(self.data_path):
            self.fallback_reason = f"Dataset '{self.data_path}' is missing from the repository."
            logger.error(self.fallback_reason)
            return

        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            
            loader = DataFrameLoader(df, page_content_column=text_col)
            documents = loader.load()

            # Fix: Using the exact stable model name without the "models/" prefix
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=api_key
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.2,
                google_api_key=api_key
            )

            self.prompt = ChatPromptTemplate.from_template(
                "You are an expert healthcare review analyst.\n"
                "Based ONLY on the provided patient reviews below, give a clear, direct, and synthesized answer to the question.\n"
                "Highlight key trends, patient sentiment, and specific details mentioned in the reviews.\n\n"
                "Context (Patient Reviews):\n{context}\n\n"
                "User Question: {question}\n\n"
                "Detailed Analysis:"
            )
            self.fallback_reason = None
            logger.info("Gemini RAG pipeline initialized successfully.")
            
        except Exception as e:
            self.fallback_reason = f"AI Initialization Crash: {str(e)}"
            logger.error(self.fallback_reason)
            self.retriever = None

    def get_response(self, user_query):
        query_clean = user_query.strip().lower()

        # Intercept casual chitchat
        if re.fullmatch(r"(hi|hello|hey|hey there|hi there|good morning|good afternoon|good evening|greetings)", query_clean):
            return (
                "👋 **Hello! Welcome to the Hospital Review Assistant.**\n\n"
                "I am an AI assistant trained on patient feedback datasets. "
                "I can analyze feedback regarding **wait times, medical care quality, staff behavior, cleanliness, and billing**.\n\n"
                "💡 **Try asking me:**\n"
                "* *'How do patients rate the medical care?'*\n"
                "* *'What are the main complaints regarding test wait times?'*\n"
                "* *'Summarize overall feedback on emergency department services.'*"
            )

        if any(w in query_clean for w in ["thanks", "thank you", "thx", "appreciate it"]):
            return "😊 You're very welcome! Feel free to ask if you need any more insights."

        if any(w in query_clean for w in ["bye", "goodbye", "see you", "cya"]):
            return "👋 Goodbye! Have a great day!"

        # Execute RAG Pipeline
        if not self.retriever or not self.llm:
            return self._local_fallback(user_query)
            
        try:
            source_docs = self.retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": user_query})
            
            answer = response.content if hasattr(response, "content") else str(response)
            
            if source_docs:
                answer += "\n\n---\n### 📑 **Referenced Patient Reviews:**\n"
                for idx, doc in enumerate(source_docs):
                    snippet = doc.page_content.replace("\n", " ").strip()
                    answer += f"* **Review {idx+1}:** \"{snippet}\"\n"
                    
            return answer
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return f"**System Error:** Unable to process the request. Details: {str(e)}"

    def _local_fallback(self, query):
        if not os.path.exists(self.data_path):
            return "❌ **Critical Error:** Dataset unavailable."
            
        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            keywords = [w for w in query.lower().split() if len(w) > 3]
            
            matches = df[df[text_col].astype(str).str.lower().apply(
                lambda x: any(k in x for k in keywords)
            )]
            
            # Print the exact diagnostic error to the UI
            answer = f"⚠️ **Fallback Mode Active**\n*Diagnostic Reason:* `{self.fallback_reason}`\n\n"
            
            if matches.empty:
                answer += "ℹ️ **No matches found.**"
                return answer
                
            top_matches = matches.head(3)[text_col].tolist()
            answer += "**Matching Reviews Found in Dataset:**\n"
            for idx, rec in enumerate(top_matches):
                answer += f"{idx+1}. *\"{rec}\"*\n\n"
                
            return answer
            
        except Exception as e:
            return "**Notice:** Could not complete local retrieval."