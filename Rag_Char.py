import os
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

            # Updated to current standard Google embedding model
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
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
        if not self.retriever or not self.llm:
            return self._local_fallback(user_query)
            
        try:
            source_docs = self.retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "question": user_query})
            
            answer = response.content if hasattr(response, "content") else str(response)
            
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