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
        """Initializes the TF-IDF retrieval engine and Gemini LLM."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY missing. AI synthesis will be disabled.")
            return
            
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset '{self.data_path}' is missing.")
            return

        try:
            self.df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in self.df.columns else self.df.columns[0]
            self.reviews = self.df[text_col].astype(str).tolist()

            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.reviews)

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0.1, # Lower temperature prevents hallucinations
                google_api_key=api_key
            )

            # UPDATED PROMPT: Forces short bullet points and related questions
            self.prompt = ChatPromptTemplate.from_template(
                "You are an expert Hospital Data Analyst.\n"
                "Your objective is to provide a clear, factual summary of patient feedback based strictly on the provided context.\n\n"
                "STRICT RULES:\n"
                "1. NO LONG PARAGRAPHS. You must answer using 2 to 3 short, concise bullet points.\n"
                "2. DO NOT quote or copy individual reviews verbatim.\n"
                "3. If the user provides only a single word (e.g., 'wait', 'cleanliness'), summarize the general sentiment regarding that specific topic.\n"
                "4. NEVER hallucinate or invent data. If the context does not contain the answer, reply: 'There is not enough patient feedback on this specific topic.'\n"
                "5. At the very end of your response, provide exactly 3 bulleted follow-up questions the user could ask, under the exact heading: '💡 Related Questions:'\n\n"
                "Patient Feedback Context:\n{context}\n\n"
                "User Query: {question}\n\n"
                "Response:"
            )
            logger.info("Pipeline initialized successfully.")
            
        except Exception as e:
            logger.error(f"AI Initialization Crash: {str(e)}")
            self.vectorizer = None

    def retrieve_context(self, query: str, top_k: int = 6) -> str:
        if self.vectorizer is None or self.tfidf_matrix is None:
            return ""
        
        query_vec = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        relevant_texts = [self.reviews[idx] for idx in top_indices]
        return "\n\n".join(relevant_texts)

    def get_response(self, user_query: Union[str, Dict]) -> str:
        if isinstance(user_query, dict):
            user_query = user_query.get("text", "")
        elif not isinstance(user_query, str):
            user_query = str(user_query)
            
        query_clean = user_query.strip().lower()

        if re.fullmatch(r"(hi|hello|hey|hey there|hi there|good morning|good afternoon|good evening|greetings)", query_clean):
            return "👋 **Hello!** I am ready to analyze patient feedback. Try sending a single keyword like 'wait' or 'cleanliness', or ask a specific question!"

        if not self.vectorizer or not self.llm:
            return "System initializing or missing API configurations."
            
        try:
            context = self.retrieve_context(user_query, top_k=6)
            if not context:
                return "I couldn't find any relevant patient feedback in the database for your query."
                
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": user_query})
            
            # STRICT FORMATTING FIX: Prevents raw JSON output
            if hasattr(response, "content"):
                answer = response.content
            elif isinstance(response, list) and len(response) > 0:
                answer = response[0].get("text", str(response))
            elif isinstance(response, dict):
                answer = response.get("text", str(response))
            else:
                answer = str(response)
                
            return answer
            
        except Exception as e:
            logger.error(f"Inference Error: {str(e)}")
            return "I am currently experiencing a connection limit. Please try asking your question again in a moment!"