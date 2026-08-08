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
        self.setup_rag()

    def setup_rag(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY missing.")
            return
            
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset '{self.data_path}' is missing.")
            return

        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            
            loader = DataFrameLoader(df, page_content_column=text_col)
            documents = loader.load()

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
            logger.info("Gemini RAG pipeline initialized successfully.")
            
        except Exception as e:
            logger.error(f"AI Initialization Crash: {str(e)}")
            self.retriever = None

    def get_response(self, user_query):
        query_clean = user_query.strip().lower()

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
            return "I am currently experiencing high traffic. Information is not available at this exact moment, but please try asking a similar question shortly!"

    def _local_fallback(self, query):
        if not os.path.exists(self.data_path):
            return "I'm sorry, but the dataset is currently unavailable."
            
        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            keywords = [w for w in query.lower().split() if len(w) > 3]
            
            matches = df[df[text_col].astype(str).str.lower().apply(
                lambda x: any(k in x for k in keywords)
            )]
            
            if matches.empty:
                return "I couldn't find any specific information related to your request. Could you please try asking a similar question regarding wait times, staff, or facilities?"
                
            top_matches = matches.head(3)[text_col].tolist()
            answer = "I am currently experiencing heavy API traffic, but here is some related feedback I found for you:\n\n"
            for idx, rec in enumerate(top_matches):
                # Clean formatting to prevent UI glitches
                clean_rec = rec.replace('\n', ' ').strip()
                answer += f"🔹 *\"{clean_rec}\"*\n\n"
                
            return answer
            
        except Exception as e:
            return "I'm sorry, I cannot retrieve the information right now. Please try again in a few moments."