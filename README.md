---
# 🏥 Hospital Review Assistant: AI-Powered Patient Feedback Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green?logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-3.6_Flash-orange?logo=google&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-5.1.0-ff69b4?logo=gradio&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-TF--IDF-blue?logo=scikitlearn&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-success?logo=githubactions)

An end-to-end, lightweight Retrieval-Augmented Generation (RAG) application that analyzes raw hospital patient reviews. It synthesizes unstructured feedback into actionable, bulleted insights for hospital administration regarding wait times, medical care, and facility cleanliness.

## 🚀 Live Demo
https://huggingface.co/spaces/MdZubair3/Hospital-Review-Bot

---

## 🧠 Architecture: 

This project intentionally bypasses heavy vector databases in favor of a lightning-fast, in-memory sparse retrieval system perfectly suited for static dataset analysis. 

Here is the step-by-step data flow:

1. **Data Ingestion:** Raw patient feedback is loaded from a local `reviews.csv` file directly into server RAM using **Pandas**.
2. **Text Vectorization:** **Scikit-Learn's `TfidfVectorizer`** converts the unstructured text into a sparse TF-IDF matrix, weighing keyword importance (e.g., "wait", "cleanliness", "staff").
3. **Semantic Retrieval:** When a user asks a question, the query is vectorized, and **Cosine Similarity** calculates the mathematical distance against the dataset, instantly retrieving the Top-K most relevant reviews.
4. **LLM Orchestration:** **LangChain** structures the retrieved context alongside the user's query into a strict prompt template. 
5. **Generative Synthesis:** The prompt is sent to the **Google Gemini (3.6 Flash)** model, which is strictly instructed to prevent hallucinations, summarize the data into 2-3 bullet points, and generate predictive follow-up questions.
6. **Frontend Delivery:** A custom-styled, monochromatic **Gradio** web interface receives the parsed output and delivers it to the user via an asynchronous, snappy chat UI.

---

## 🛠️ Tech Stack & Tools

* **Core Language:** Python
* **LLM Provider:** Google Gemini API (`gemini-3.6-flash`)
* **AI Orchestration:** LangChain
* **Data Science & Retrieval:** Scikit-Learn (TF-IDF, Cosine Similarity), Pandas
* **Frontend UI:** Gradio (with custom CSS for a premium SaaS aesthetic)
* **CI/CD:** GitHub Actions (Automated Pytest pipelines)
* **Deployment:** Hugging Face Spaces (Dockerized)

---

## ✨ Key Features

* **Serverless RAG Pipeline:** Achieves high-accuracy information retrieval without the overhead of external database connections.
* **Strict Prompt Engineering:** The LLM is restricted from generating long paragraphs or hallucinating data outside the provided CSV context.
* **Smart UI Routing:** Implements custom queue management in Gradio for instant UI updates, providing a snappy, responsive user experience.
* **Premium Dark Mode UI:** Features a custom charcoal CSS theme with locked resizing, tailored scrollbars, and tactile buttons.
* **Automated CI/CD Testing:** Integrated with GitHub Actions to run environment tests on every push.

---

