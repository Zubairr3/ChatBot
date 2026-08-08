import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
        openai_key = os.environ.get("OPENAI_API_KEY")

        if not openai_key:
            logger.warning("OPENAI_API_KEY not found. System will default to local fallback retrieval.")
            return

        if not os.path.exists(self.data_path):
            logger.error(f"Dataset '{self.data_path}' is missing.")
            return

        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            
            loader = DataFrameLoader(df, page_content_column=text_col)
            documents = loader.load()

            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            vectorstore = FAISS.from_documents(documents, embeddings)

            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_key)

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            logger.info("Production RAG pipeline initialized successfully.")
            
        except Exception as e:
            logger.error(f"RAG Initialization Error: {e}")
            self.qa_chain = None

    def get_response(self, user_query):
        if not self.qa_chain:
            return self._local_fallback(user_query)

        try:
            response = self.qa_chain.invoke({"query": user_query})
            answer = response.get("result", "I couldn't find relevant information for that query.")

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
            return "**Notice:** System is operating offline and local data is unavailable."

        try:
            df = pd.read_csv(self.data_path)
            text_col = "review" if "review" in df.columns else df.columns[0]
            
            keywords = query.lower().split()
            matches = df[df[text_col].astype(str).str.lower().apply(lambda x: any(k in x for k in keywords))]

            if matches.empty:
                return "**Fallback Mode:** No API key provided, and local keyword search yielded no relevant reviews."

            answer = "**Fallback Mode (API Key Missing):** Displaying top local keyword match.\n\n"
            answer += f"**Matched Record:** {matches.iloc[0][text_col]}\n"
            return answer
            
        except Exception as e:
            logger.error(f"Fallback Search Error: {e}")
            return "**Notice:** System operating offline. Could not complete local retrieval."