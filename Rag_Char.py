import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HospitalReviewBot:
    def __init__(self, data_path="reviews.csv"):
        self.data_path = data_path
        self.qa_chain = None
        self.setup_rag()

    def setup_rag(self):
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not google_api_key:
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

            # Initialize Gemini Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=google_api_key
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Initialize Gemini Chat Model
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0, 
                google_api_key=google_api_key
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            logger.info("Gemini RAG pipeline initialized.")
            
        except Exception as e:
            logger.error(f"Initialization Error: {e}")
            self.qa_chain = None

    def get_response(self, user_query):
        if not self.qa_chain:
            return self._local_fallback(user_query)
            
        try:
            response = self.qa_chain.invoke({"query": user_query})
            answer = response.get("result", "I couldn't find relevant information.")
            
            source_docs = response.get("source_documents", [])
            if source_docs:
                answer += "\n\n### **Source Citations:**\n"
                for idx, doc in enumerate(source_docs):
                    snippet = doc.page_content[:100].replace("\n", " ").strip()
                    answer += f"* **Source {idx+1}:** \"{snippet}...\"\n"
                    
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
                return "**Fallback Mode:** No API key provided; local search yielded no results."
                
            answer = "**Fallback Mode (API Key Missing):** Displaying top local match.\n\n"
            answer += f"**Matched Record:** {matches.iloc[0][text_col]}\n"
            return answer
            
        except Exception as e:
            return "**Notice:** Could not complete local retrieval."