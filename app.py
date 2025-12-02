import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



base = "https://www.mygreatlearning.com"
base_url = base + "/academy" 

st.title("🎓 GreatLearning Course Assistant")

st.markdown("""
Welcome to your **AI-powered GreatLearning companion**.  
Search courses, explore career paths, and ask questions — all supported by a smart RAG-based engine that learns from real course pages.
""")




# -------------------------------------------
# GOOGLE API KEY
# -------------------------------------------
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ Google API Key not found!")
    st.stop()



# -------------------------------------------
# SIDEBAR
# -------------------------------------------
with st.sidebar:
    category = st.multiselect("Select Course Category:", ['Free Courses', 'Premium Courses', 'Career Path'])
    st.markdown("---")
    
    keywords_list = st.text_input("Enter a course / career path :")
    keywords = keywords_list.split()
    
    clear_button = st.button("🧹 Clear Results")

    if clear_button:
        keys_to_clear = ["url_links", "metadata", "vectordb"]

        for key in keys_to_clear:
            if key in st.session_state: 
                del st.session_state[key]

        st.success("Search results cleared!")
        st.stop()
    
    category_list_info = []

    for item in category:
        current_url = base_url 

        if item == 'Career Path':
            current_url += "/career-paths/"
            category_list_info.append((current_url, ['/career-paths'] , 'Career Path'))
            
        elif item == 'Premium Courses':
            current_url += "/premium/"
            category_list_info.append((current_url, ['/premium'] , 'Premium Course'))
            
        elif item == 'Free Courses':
            current_url += "/learn-for-free/"
            if current_url == '/free-courses':
                continue
            free_course_keywords = ['/learn-for-free/courses/']
            category_list_info.append((current_url, free_course_keywords , 'Free Course'))

# ------------------------------------------------
# SCRAPE COURSES
# ------------------------------------------------
def scrape_courses(category_info_list, user_keywords):
    unique_link_with_info = []
    try:
        for scrape_url, path_keywords, course_type in category_info_list:

            response = requests.get(scrape_url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            course_links = set()
            
            for a in soup.find_all("a"):
                href = a.get("href")
                if not href:
                    continue
                
                is_category_link = any(kw in href for kw in path_keywords)  
                if course_type == "Career Path":
                    slug = href.rstrip("/").split("/")[-1].lower()
                    modified_keywords = ["-".join(user_keywords).lower()] 
                    is_keyword_match = slug == modified_keywords[0]
                else:
                    is_keyword_match = any(kw.lower() in href.lower() for kw in user_keywords)
                

                if is_category_link and is_keyword_match:
                    full_url = base + href if href.startswith("/") else href
                    substring = full_url.split('?')[0]
                    course_links.add(substring)

            if course_links:
                for link in course_links:
                    unique_link_with_info.append((link , course_type))
            
            
        if unique_link_with_info:
            # st.markdown(unique_links)
            build_knowledge_base(unique_link_with_info)
        else:
            st.info(f"No courses found in this category matching: {', '.join(user_keywords)}")
            return "Not Valid"

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ------------------------------------------------
# BUILD KNOWLEDGE BASE
# ------------------------------------------------
def build_knowledge_base(unique_link_with_info):

    if "url_links" not in st.session_state:
        st.session_state.url_links = [url for url, _ in unique_link_with_info]

    if "metadata" not in st.session_state:
        st.session_state.metadata = extract_metadata(unique_link_with_info)

    metadata = st.session_state.metadata
    if metadata:
        st.markdown(f'Found {len(metadata)} relevant data sources.')
        with st.expander("All Courses Summary", expanded=False):
            for course in metadata:
                display_course(course)

    if "vectordb" not in st.session_state:
        with st.spinner("Building vector database..."):
            docs = load_and_process_documents(st.session_state.url_links)
            st.session_state.vectordb = build_vectordb(docs)

    vectordb = st.session_state.vectordb

    st.markdown("---")



        
# ---------------------------------------------------
# EXTRACT METADATA
# ---------------------------------------------------
def extract_metadata(urls):
    course_data = []
    with st.spinner("Loading metadata..."):

        for url , course_type in urls:
            try:
                page = requests.get(url=url, timeout=10)
                soup = BeautifulSoup(page.text, "html.parser")
                
                course_link = url

                if course_type == 'Free Course' or course_type == 'Premium Course':
                    title = duration = level = learners = projects = ratings = None

                    title_tag = soup.find("h2", class_ ="header-container__title")
                    title = title_tag.text.strip() if title_tag else 'Not Found'

                    details_container = soup.find('div', class_=lambda c: c and 'align-items-center' in c and 'flex-wrap' in c)
                    
                    details  = []
                    rating_icon = soup.find("img", class_="course-summary__icon star")
                    ratings = None
                    if rating_icon:
                        rating_span = rating_icon.find_next("span", class_="new_course_dot")
                        if rating_span:
                            ratings = rating_span.get_text(strip=True).replace("\xa0", "")
                    if details_container:
                        for span in details_container.find_all("span", class_=lambda c: c and "new_course_dot" in c):
                            text = span.get_text(strip=True)
                            if any(keyword in text.lower() for keyword in ["hours", "hrs", "hr" , "learning hrs"]):
                                duration = text
                                continue
                            if any(keyword in text.lower() for keyword in ["level"]):
                                level = text
                                continue
                            if any(keyword in text.lower() for keyword in ["learners"]):
                                learners = text
                                continue
                            if any(keyword in text.lower() for keyword in ["projects" , "project"]):
                                projects = text
                                continue
                            elif text:
                                details.append(text)
                            

                    # details_text = " | ".join(details) if details else "Details Not Found"
                    
                    course_data.append({
                            "Link" : course_link,
                            "Title" : title,
                            "Course Type" : course_type,
                            "Duration" : duration,
                            "Level" :  level,
                            "Learners" : learners,
                            "Projects" : projects,
                            "Ratings": ratings,
                            "Details": details
                        })
                else:
                    title = desc =  None
                    title_tag = soup.find("h1", class_ ="career-path-name")
                    title = title_tag.text.strip() if title_tag else 'Not Found'
                    desc_tag = soup.find("div", class_ = "career-path-desc")
                    desc = desc_tag.get_text(strip=True) if desc_tag else 'Not Found'
                    course_data.append({
                        "Title": title,
                        "Description" : desc,
                        "Course Type": course_type,
                        "Link" : course_link
                    })

            except Exception as e:
                st.markdown(f"Error extracting details from {url}: {e}")
    return course_data

# ------------------------------------------------
# DISPLAY COURSE
# ------------------------------------------------
def display_course(course):
    fields = {
        "⭐ Ratings": course.get("Ratings"),
        "⏳ Duration": course.get("Duration"),
        "🎯 Level": course.get("Level"),
        "🧩 Projects": course.get("Projects"),
        "👨‍🎓 Learners": course.get("Learners"),
        "📚 Courses": course.get("Courses"),
        "📝 Description": course.get("Description"),
    }


    f"""**{course.get('Title', 'No Title')} ({course.get("Course Type")})**"""

    # Add only available fields
    for label, value in fields.items():
        if value not in [None, "", "Not Found", [], "N/A"]:
            f""" {label} : {value} """

    link = course.get("Link")
    if link:
        f"""{link}"""

    "-" * 60

# ------------------------------------------------
# LOAD AND PROCESS DOCUMENTS
# ------------------------------------------------
def load_and_process_documents(url_links):
    docs = []
    progress = st.progress(0)
    status = st.empty()

    for i, u in enumerate(url_links):
        status.text(f"⏳ Loading course page {i+1}/{len(url_links)}")
        try:
            loader = WebBaseLoader(web_paths=[u])
            docs.extend(loader.load())
        except:
            st.warning(f"Failed to load: {u}")

        progress.progress((i+1)/len(url_links))

    status.empty()
    progress.empty()
    return docs

# ------------------------------------------------
# BUILD VECTOR DB
# ------------------------------------------------
def build_vectordb(docs):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(chunks, embeddings)
        return vectordb
    except Exception as e:
        st.error(f"Error building vector DB: {e}")


# ------------------------------------------------
# GENERATE ANSWER
# ------------------------------------------------
def generate_answer(query, vectordb):
    with st.spinner("`Generating answer ..`"):

        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        
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

        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=GOOGLE_API_KEY,
                temperature=0.3,
            )
            
            def format_docs(docs):
                text = "\n\n".join(d.page_content for d in docs)
                return text[:2000] + "..." if len(text) > 2000 else text
            
            rag_chain = (
                {
                    "context": retriever| format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(query)
        except Exception as e:
            st.markdown(f"Error with LLM. Error generating answer.{e}")

    st.subheader("✨ Answer")
    st.write(answer)


# ------------------------------------------------
# Main Content
# ------------------------------------------------
if category_list_info and keywords:
    is_valid_search = scrape_courses(category_list_info, keywords)
    if is_valid_search != "Not Valid":
        query = st.text_input("🔎 Ask your question:", placeholder="e.g., Which course has the highest rating?") 
        if query and "vectordb" in st.session_state: 
            generate_answer(query, st.session_state.vectordb)
elif not category_list_info:
    st.info("Please select at least one category from the sidebar.")
elif not keywords:
    st.info("Please enter a course topic you want to search in the sidebar.")
