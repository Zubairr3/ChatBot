import os
import pandas as pd
import logging
import re
from typing import Union, Dict

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

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
        """Initializes RAG with local embeddings and strict abstractive summarization prompt."""
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

            # 2. Setup Local Embeddings (Zero startup API cost)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(documents, embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            
            # 3. Setup Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.3,
                google_api_key=api_key
            )

            # 4. Strict Abstractive Summarization Prompt (Prevents raw CSV copying)
            self.prompt = ChatPromptTemplate.from_template(
                "You are an expert Hospital Clinical Quality Director and Data Analyst.\n"
                "Your sole objective is to write an abstractive, professional EXECUTIVE SUMMARY that directly answers the user's question based on the provided patient feedback.\n\n"
                "STRICT RULES:\n"
                "1. DO NOT quote, copy, or paste individual reviews verbatim.\n"
                "2. DO NOT list individual customer review texts or use quotation marks for raw feedback.\n"
                "3. Synthesize the core themes, overall sentiment, trends, and patterns across the feedback into concise, professional paragraphs.\n\n"
                "Patient Feedback Context:\n{context}\n\n"
                "User Question: {question}\n\n"
                "Executive Summary:"
            )
            logger.info("RAG pipeline initialized successfully with strict summarization rules.")
            
        except Exception as e:
            logger.error(f"AI Initialization Crash: {str(e)}")
            self.retriever = None

    def get_response(self, user_query: Union[str, Dict]) -> str:
        """Processes the user query and returns a synthesized AI summary."""
        
        if isinstance(user_query, dict):
            user_query = user_query.get("text", "")
        elif not isinstance(user_query, str):
            user_query = str(user_query)
            
        query_clean = user_query.strip().lower()

        # Polite Chitchat & Intent Detection
        if re.fullmatch(r"(hi|hello|hey|hey there|hi there|good morning|good afternoon|good evening|greetings)", query_clean):
            return (
                "👋 **Hello! Welcome to the Hospital Review Assistant.**\n\n"
                "I am an AI assistant trained to analyze and summarize patient feedback regarding **wait times, medical care quality, staff behavior, cleanliness, and billing**.\n\n"
                "💡 **Try asking me:**\n"
                "* *'How do patients rate the medical care?'*\n"
                "* *'Summarize the general feedback regarding hospital cleanliness.'*"
            )

        if any(w in query_clean for w in ["thanks", "thank you", "thx", "appreciate it"]):
            return "😊 You're very welcome! Feel free to ask if you need any more synthesized insights."

        if any(w in query_clean for w in ["bye", "goodbye", "see you", "cya"]):
            return "👋 Goodbye! Have a great day!"

        if not self.retriever or not self.llm:
            return "I am currently initializing my analysis engine or missing API configurations. Please check your system settings."
            
        # Execute AI RAG Pipeline with Strict Summary Prompt
        try:
            source_docs = self.retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": user_query})
            
            answer = response.content if hasattr(response, "content") else str(response)
            return answer
            
        except Exception as e:
            logger.error(f"Inference Error: {str(e)}")
            return "I am currently experiencing a temporary connection limit. Please try asking your question again in a moment!"