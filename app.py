import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# ---------------------------------------------------
#  Hidden HuggingFace Token
# ---------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="GreatLearning RAG Bot",
    page_icon="🎓",
    layout="wide"
)

# Main Title + Description
st.title("🎓 GreatLearning Academy RAG Assistant")
st.markdown("""
This tool allows you to explore **GreatLearning Academy courses** using a Retrieval-Augmented Generation (RAG) chatbot.  
It automatically scrapes course pages, builds a knowledge base, and answers your course-related questions with AI.
""")

# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("Manage your scraping and knowledge base settings below.")

    max_courses = st.number_input(
        "📑 Number of Courses to Scrape",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    build_kb = st.button("💡 Build Knowledge Base")

    st.markdown("---")
    st.info("After building the knowledge base, scroll down to ask course-related questions.")
    st.markdown("---")

# ---------------------------------------------------
# SCRAPER
# ---------------------------------------------------
@st.cache_data
def get_course_links(max_courses=10):
    base = "https://www.mygreatlearning.com"
    url = base + "/academy"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/academy/learn-for-free/courses" in href:
            if href.startswith("/"):
                href = base + href
            links.append(href) 

    return list(set(links))[:max_courses]

# ---------------------------------------------------
# LOADING AND PROCESSING
# ---------------------------------------------------
def load_and_process_docs(urls):
    docs = []
    progress = st.progress(0)
    status = st.empty()

    for i, u in enumerate(urls):
        status.text(f"⏳ Loading course page {i+1}/{len(urls)}")
        try:
            loader = WebBaseLoader(web_paths=[u])
            docs.extend(loader.load())
        except:
            st.warning(f"Failed to load: {u}")

        progress.progress((i+1)/len(urls))

    status.empty()
    progress.empty()
    return docs

# ---------------------------------------------------
# VECTOR DB
# ---------------------------------------------------
@st.cache_resource
def build_vectordb(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(chunks, embeddings)

# ---------------------------------------------------
# BUILD KNOWLEDGE BASE PROCESS
# ---------------------------------------------------
if build_kb:

    with st.spinner("Scraping course pages..."):
        urls = get_course_links(max_courses)

    st.success("Course URLs successfully scraped!")

    st.session_state["urls"] = urls

    with st.spinner("Loading course content..."):
        docs = load_and_process_docs(urls)


    with st.spinner("Creating vector DB..."):
        vectordb = build_vectordb(docs)

    st.session_state["retriever"] = vectordb.as_retriever(search_kwargs={"k": 2})

    st.success("Your knowledge Base is ready! You can now ask questions below.")

# ---------------------------------------------------
# CHAT INTERFACE
# ---------------------------------------------------
if "retriever" in st.session_state:

    st.markdown("## 💬 Ask Questions About Courses")
    st.markdown("""
    Type any question about the scraped courses below.
     It will answer using only the information extracted from the course pages.
    """)

    # ---------------------------------------------------
    # SHOW COURSE LIST
    #  ---------------------------------------------------
    if "urls" in st.session_state:
        with st.expander("🔗 View scraped Course URLs", expanded=False):
            for u in st.session_state["urls"]:
                st.markdown(f"[{u}]({u})")


    query = st.text_input("🔎 Enter your question:", placeholder="e.g., Does the course provide a certificate?")

    if query:
        with st.spinner("🤖 Generating answer..."):

            prompt_template = """
                You are an expert assistant for question answering about GreatLearning Academy courses. 

                You are given: 
                1. Context information from course pages 
                2. A question from the user 

                Instructions: 
                - Use ONLY the given context to answer. 
                - If the answer is not in the context, say: 
                    "I don't have information about this in the course content." 
                - Keep the answer clear, concise, and helpful. 
                - Be friendly and professional. 

                Context: {context} 

                Question: {question} 
                Answer:
                """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            base_llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.3,
                max_new_tokens=300
            )
            
            llm = ChatHuggingFace(llm=base_llm)

            def format_docs(docs):
                text = "\n\n".join(d.page_content for d in docs)
                return text[:2000] + "..." if len(text) > 2000 else text

            rag_chain = (
                {
                    "context": st.session_state["retriever"] | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(query)

        st.subheader("✨ Answer")
        st.write(answer)

# ------------------------#
#  Footer
# ------------------------#
st.markdown("---")
st.caption(
    "Built with ❤️ using Streamlit, LangChain, HuggingFace & ChromaDB | Data Source: My Great Learning  | Developer: Mishalee Lambat"
)

