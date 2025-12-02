# 🎓 GreatLearning RAG Chatbot

A **Streamlit-based Retrieval-Augmented Generation (RAG) pipeline** for exploring **GreatLearning Academy courses**. This app scrapes course pages, builds a vector database, and allows AI-powered question answering.

🔗 **Live Demo:** [View on Streamlit](#)  <!-- Replace with your actual link -->

---

## 🚀 Project Overview

This project is an **end-to-end RAG pipeline** designed to help users query **GreatLearning Academy course content**:

- 🔍 **Course Scraper** – Extracts course titles, descriptions, URLs, duration, ratings, learners, and projects.
- 🧹 **Text Cleaning & Preprocessing** – Cleans HTML, normalizes whitespace, and prepares text for embedding.
- 📄 **Vector Store Indexing** – Uses **HuggingFace sentence-transformers/all-MiniLM-L6-v2** for embeddings.
- 🤖 **LLM-based Question Answering** – Uses **Google Gemini API** for AI-powered answers.
- 🌐 **Interactive Streamlit UI** – Sidebar for scraping and keyword selection, main panel for queries and results.

---

## 🛠 Features

✔️ **Flexible Course Scraping**  
- Search by **Free Courses**, **Premium Courses**, or **Career Paths**.  
- Automatically converts career path queries to **hyphenated URLs**.  
- Stores previously scraped results in **session state**.  
- Properly clickable course links in the Streamlit app.

✔️ **Metadata Extraction**  
- Title, description, duration, level, learners, projects, ratings.  
- Handles both **courses and career paths** separately.

✔️ **RAG Query System**  
- Scraped course content is split into **chunks** for embeddings.  
- User question retrieves relevant course chunks.  
- Generates answers **grounded in scraped course content**.  

✔️ **Clean UI**  
- Left sidebar for selecting course type & entering keywords.  
- Main area for displaying course summaries & interactive Q&A.  
- "Clear Results" button to reset previous searches.  

---

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

GOOGLE_API_KEY=your_google_api_key_here


### 4. Run App
    streamlit run app.py

## 🎯 How It Works
### 1. Select & Scrape Courses
 - Choose Free Courses, Premium Courses, or Career Paths.
 - Enter keywords (e.g., Machine Learning Engineer).
 - Career path keywords automatically use hyphens for URL search.

### 2. Metadata Extraction
 - Scrapes course titles, descriptions, durations, levels, learners, projects, ratings, and links.
 - Stores results in Streamlit session state.

### 3. Build Vector Database
 - Extracted text is split into chunks using RecursiveCharacterTextSplitter.
 - Converted into embeddings via sentence-transformers/all-MiniLM-L6-v2.
 - Stored in Chroma vector DB for retrieval.

### 4. AI-Powered Question Answering
 - User enters a question in the main panel.
 - Relevant course chunks retrieved from the vector DB.
 - Google Gemini LLM generates concise, grounded answers.

## 🛠 Technologies Used
  - Python 3.10+
  - Streamlit
  - BeautifulSoup / Requests
  - HuggingFace Sentence Transformers
  - Chroma vector store
  - Google Gemini API
  - dotenv

## 📦 Future Enhancements
 - Multi-provider scraping (Coursera, Udemy, edX).
 - Improved caching for faster vector DB build.
 - User authentication for personalized queries.
 - Dashboard analytics for course recommendations.
