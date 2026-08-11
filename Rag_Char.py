import os
import pandas as pd
import logging
import re
from typing import Union, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HospitalReviewBot:
    def __init__(self, data_path: str = "reviews.csv"):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.llm = None
        self.prompt = None
        self.setup_rag()

    def setup_rag(self) -> None:
        """Initializes a lightweight TF-IDF retrieval engine and Gemini LLM."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY missing. AI synthesis will be disabled.")
            return
            
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset '{self.data_path}' is missing.")
            return

        try:
            # 1. Load Data
            self.df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in self.df.columns else self.df.columns[0]
            self.reviews = self.df[text_col].astype(str).tolist()

            # 2. Build Lightweight TF-IDF Matrix (Lightning fast, zero PyTorch overhead)
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.reviews)

            # 3. Setup Gemini LLM for Generation (Updated to -latest to fix 404 API error)
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", 
                temperature=0.3,
                google_api_key=api_key
            )

            # 4. Strict Abstractive Summarization Prompt
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
            logger.info("Lightweight TF-IDF retrieval & Gemini RAG pipeline initialized successfully.")
            
        except Exception as e:
            logger.error(f"AI Initialization Crash: {str(e)}")
            self.vectorizer = None

    def retrieve_context(self, query: str, top_k: int = 6) -> str:
        """Finds the most relevant reviews using cosine similarity on TF-IDF vectors."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return ""
        
        query_vec = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        relevant_texts = [self.reviews[idx] for idx in top_indices]
        return "\n\n".join(relevant_texts)

    def get_response(self, user_query: Union[str, Dict]) -> str:
        """Processes user query and returns a synthesized AI summary."""
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

        if not self.vectorizer or not self.llm:
            return "I am currently initializing my analysis engine or missing API configurations. Please check your system settings."
            
        try:
            context = self.retrieve_context(user_query, top_k=6)
            if not context:
                return "I couldn't find any relevant patient feedback in the database for your query."
                
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": user_query})
            
            answer = response.content if hasattr(response, "content") else str(response)
            return answer
            
        except Exception as e:
            logger.error(f"Inference Error: {str(e)}")
            return "I am currently experiencing a temporary connection limit. Please try asking your question again in a moment!"

if __name__ == "__main__":
    bot = HospitalReviewBot(data_path="reviews.csv")
    test_query = "What do patients say about wait times for tests?"
    print(bot.get_response(test_query))

    test_query = "How do patients rate the medical care?"
    print(bot.get_response(test_query))

    test_query = "Summarize general feedback regarding hospital cleanliness."
    print(bot.get_response(test_query))