import os
import pandas as pd
import logging
import re
from typing import Union, Dict

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Configure enterprise-grade logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HospitalReviewBot:
    def __init__(self, data_path: str = "reviews.csv"):
        self.data_path = data_path
        self.retriever = None
        self.llm = None
        self.prompt = None
        self.setup_rag()

    def setup_rag(self) -> None:
        """Initializes the RAG pipeline with Google Gemini and FAISS Vector Database."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY missing. AI synthesis will be disabled.")
            return
            
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset '{self.data_path}' is missing.")
            return

        try:
            # 1. Load Data
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            
            loader = DataFrameLoader(df, page_content_column=text_col)
            documents = loader.load()

            # 2. Setup Vector Database
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=api_key
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # 3. Setup Large Language Model (LLM)
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.2,
                google_api_key=api_key
            )

            # 4. Configure Prompt Architecture
            self.prompt = ChatPromptTemplate.from_template(
                "You are an expert healthcare review analyst.\n"
                "Based ONLY on the provided patient reviews below, give a clear, direct, and synthesized answer to the question.\n"
                "Highlight key trends, patient sentiment, and specific details mentioned in the reviews.\n\n"
                "Context (Patient Reviews):\n{context}\n\n"
                "User Question: {question}\n\n"
                "Detailed Analysis:"
            )
            logger.info("Gemini RAG pipeline initialized successfully.")
            
        except Exception as e:
            logger.error(f"AI Initialization Crash: {str(e)}")
            self.retriever = None

    def get_response(self, user_query: Union[str, Dict]) -> str:
        """Processes the user query and returns a synthesized AI response or polite fallback."""
        
        # Robust Input Sanitization (Protects against Gradio 5 Dict payloads)
        if isinstance(user_query, dict):
            user_query = user_query.get("text", "")
        elif not isinstance(user_query, str):
            user_query = str(user_query)
            
        query_clean = user_query.strip().lower()

        # 1. Polite Chitchat & Intent Detection
        if re.fullmatch(r"(hi|hello|hey|hey there|hi there|good morning|good afternoon|good evening|greetings)", query_clean):
            return (
                "👋 **Hello! Welcome to the Hospital Review Assistant.**\n\n"
                "I am an AI assistant trained to analyze patient feedback regarding **wait times, medical care quality, staff behavior, cleanliness, and billing**.\n\n"
                "💡 **Try asking me:**\n"
                "* *'How do patients rate the medical care?'*\n"
                "* *'What are the main complaints regarding test wait times?'*"
            )

        if any(w in query_clean for w in ["thanks", "thank you", "thx", "appreciate it"]):
            return "😊 You're very welcome! Feel free to ask if you need any more insights."

        if any(w in query_clean for w in ["bye", "goodbye", "see you", "cya"]):
            return "👋 Goodbye! Have a great day!"

        # 2. Check if RAG is successfully loaded
        if not self.retriever or not self.llm:
            return self._local_fallback(user_query)
            
        # 3. Execute AI RAG Pipeline
        try:
            source_docs = self.retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": user_query})
            
            answer = response.content if hasattr(response, "content") else str(response)
            
            # Format Citations cleanly
            if source_docs:
                answer += "\n\n---\n### 📑 **Referenced Patient Reviews:**\n"
                for idx, doc in enumerate(source_docs):
                    snippet = doc.page_content.replace("\n", " ").strip()
                    answer += f"* **Review {idx+1}:** \"{snippet}\"\n"
                    
            return answer
            
        except Exception as e:
            logger.error(f"Inference Error: {str(e)}")
            return "I am currently experiencing high traffic or a temporary API limitation. Information is not available at this exact moment, but please try asking a similar question shortly!"

    def _local_fallback(self, query: str) -> str:
        """Executes a local keyword search when the API is unavailable."""
        if not os.path.exists(self.data_path):
            return "I'm sorry, but the dataset is currently unavailable."
            
        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            
            # Extract meaningful keywords (length > 3)
            keywords = [w for w in query.lower().split() if len(w) > 3]
            
            if not keywords:
                return "Could you please provide a more specific question regarding the hospital?"
                
            matches = df[df[text_col].astype(str).str.lower().apply(
                lambda x: any(k in x for k in keywords)
            )]
            
            if matches.empty:
                return "I couldn't find any specific information related to your request. Could you please try asking a similar question regarding wait times, staff, or facilities?"
                
            top_matches = matches.head(3)[text_col].tolist()
            answer = "I am currently experiencing heavy API traffic, but here is some related feedback I found for you:\n\n"
            for idx, rec in enumerate(top_matches):
                clean_rec = rec.replace('\n', ' ').strip()
                answer += f"🔹 *\"{clean_rec}\"*\n\n"
                
            return answer
            
        except Exception as e:
            logger.error(f"Fallback Error: {str(e)}")
            return "I'm sorry, I cannot retrieve the information right now. Please try again in a few moments."