# 🎓 Great Learning RAG Pipeline/Chatbot
A Streamlit-based RAG (Retrieval-Augmented Generation) pipeline/chatbot that scrapes GreatLearning Academy course pages, builds a vector database, and answers questions using LLMs.

 🔗 **Live Demo:** [View on Streamlit](https://great-learning-rag-chatbot-qvhftpkzxhngmkqeygfrzp.streamlit.app/)
  

## 🚀 Project Overview

This project is an end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to help users explore and query online course content.
It includes:
  - 🔍 Course Scraper – Extracts course titles, URLs, descriptions, and ratings
  - 🧹 Text Cleaning & Preprocessing
  - 📄 Vector Store Indexing using Sentence Transformers
  - 🤖 LLM-based Question Answering using HuggingFace Inference API
  - 🌐 Streamlit Web App – Clean UI to scrape, store, and query courses

## 🚀 Features
✔️ Scrape Courses 
  - Fetch up to 10+ courses from any valid Udemy/Coursera category URL
  - Keeps previously scraped list visible—no flash/reset
  - Proper URL clickability within Streamlit

✔️ Clean & Prepare Data
  - Removes HTML tags, emojis, special chars
  - Normalizes whitespace
  - Simple text preprocessing pipeline

✔️ Vectorization
  - Uses all-MiniLM-L6-v2 for embedding

✔️ LLM-Powered Query
  - For any user question, system fetches best chunks from vector store
  - Sends them to HuggingFace Inference API
  - Uses environment variable for API key

✔️ Fully Interactive UI
  - Left sidebar for scraping
  - Main area for search
  - Clean responsive layout

🧩 Project Structure

    course-rag-pipeline/
    │
    ├── app.py                     # Main Streamlit application
    ├── requirements.txt           # Python dependencies
    └── README.md                  # Project documentation


## 🔧 Installation & Setup
### 1. Clone Repository

    git clone https://github.com/<your-username>/course-rag-pipeline.git
    cd course-rag-pipeline

### 2. Install Dependencies
    pip install -r requirements.txt


### 3. Add Environment Variable

Create a .env file:

HUGGINGFACEHUB_API_TOKEN=hf_XXXXXXXXXXXXXXXXXXXX


### 4. Run App
    streamlit run app.py

## 🎯 How It Works
### 1. Scraping  
- User enters a course category URL → Scraper extracts all course metadata.

### 2. Preprocessing
- Text gets cleaned and simplified.

### 3. Embedding

- System converts course descriptions into vector embeddings.

### 4. RAG Querying

- User asks a question → System retrieves best-matched courses → LLM generates an answer grounded in local data.

## 🧪 Example Use Cases

    "Suggest courses for beginner data analysts"

    "List the courses that include hands-on exercises"

    "Which courses are best for absolute beginners?"

    "Summarize all Python-related courses"

## 🛠 Technologies Used
  - Python 3.10+
  - Streamlit
  - BeautifulSoup / Requests
  - SentenceTransformers
  - FAISS
  - HuggingFace Inference API
  - dotenv

## 📦 Future Enhancements
- Add multi-provider scraping (Udemy + Coursera + edX)
