import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

# Setup LLM
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-0528:free",
    temperature=0.9,
    openai_api_key=api_key,
    openai_api_base=api_base,
)

# Streamlit UI Config
st.set_page_config(layout="wide")
st.markdown("<style>body { overflow-x: hidden; }</style>", unsafe_allow_html=True)
st.title("📘 AI Article Research Tool")

# Session state
if "url_count" not in st.session_state:
    st.session_state.url_count = 1
if "urls" not in st.session_state:
    st.session_state.urls = [""]
if "result" not in st.session_state:
    st.session_state.result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Layout
left_col, right_col = st.columns([0.85, 2])

# ---------------- LEFT COLUMN ----------------
with left_col:
    st.subheader("🔗 Add Article URLs")
    temp_urls = []

    for i in range(st.session_state.url_count):
        col_url, col_btn = st.columns([0.9, 0.1])
        with col_url:
            url = st.text_input(f"Article URL {i+1}", key=f"url_{i}", value=st.session_state.urls[i])
            temp_urls.append(url)
        with col_btn:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.urls.pop(i)
                st.session_state.url_count -= 1
                st.rerun()

    if st.button("➕ Add Another URL"):
        st.session_state.url_count += 1
        st.session_state.urls.append("")
        st.session_state.result = None

    if st.button("🔍 Analyze Articles"):
        valid_urls = [url for url in temp_urls if url.strip() != ""]
        if not valid_urls:
            st.warning("Please enter at least one valid URL.")
        else:
            loader = UnstructuredURLLoader(urls=valid_urls)
            st.info("📥 Loading data from URLs...")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap=150
            )
            st.info("📚 Splitting text into manageable chunks...")
            docs = text_splitter.split_documents(data)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.info("🔧 Creating embedding vector store...")
            time.sleep(2)

            with open("faiss_store_deepseek.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

            st.session_state.vectorstore_created = True
            st.success("✅ Articles processed. You can now ask a question.")

# ---------------- RIGHT COLUMN ----------------
with right_col:
    st.subheader("❓ Ask Your Research Question")
    query = st.text_input("Ask a question", key="question_input", label_visibility="collapsed")

    col1, col2 = st.columns([1.2, 0.1])
    with col1:
        submit_query = st.button("Submit Question")
    with col2:
        reset_query = st.button("🔄 Reset")

    if reset_query:
        st.session_state.result = None
        st.session_state.last_query = ""
        st.rerun()

    if submit_query and query and query != st.session_state.last_query:
        if os.path.exists("faiss_store_deepseek.pkl"):
            with open("faiss_store_deepseek.pkl", "rb") as f:
                vectorstore = pickle.load(f)
                retriever = vectorstore.as_retriever()

            # Custom clean prompt
            custom_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are an AI research assistant. Use the following context to generate a thorough, well-structured answer to the research question.

Instructions:
- Start with a brief introduction to the topic.
- Follow up with detailed bullet points.
- Use plain language.
- Do NOT use symbols like '*', '#', or special markdown. Only use bullet points and paragraph text.
- End with a short concluding insight.

Context:
{context}

Question:
{question}

Answer:
"""
            )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": custom_prompt}
            )

            try:
                st.session_state.result = chain.invoke({"query": query})
                st.session_state.last_query = query
            except Exception as e:
                st.error(f"❌ Error during API call: {e}")

    if st.session_state.result:
        st.header("📑 Research Answer")
        answer = st.session_state.result.get("answer") or st.session_state.result.get("result")
        if answer:
            cleaned_answer = answer.replace("*", "").replace("#", "").replace(">", "")
            st.markdown(cleaned_answer, unsafe_allow_html=True)

        sources = st.session_state.result.get("source_documents", [])
        if sources:
            st.subheader("🔗 Sources")
            for i, doc in enumerate(sources):
                source_url = doc.metadata.get("source", "")
                if source_url:
                    st.markdown(f"- [Source {i+1}]({source_url})", unsafe_allow_html=True)
