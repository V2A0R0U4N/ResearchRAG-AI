import os
import streamlit as st
import pickle
import time
import re
from dotenv import load_dotenv

# LangChain & Community Imports
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Internal Import
from query_guard import is_vague_query, check_relevance_with_scores

load_dotenv()

# ========== Streamlit UI Config ==========
st.set_page_config(layout="wide", page_title="Advanced ResearchRAG-AI")
st.markdown("<style>body { overflow-x: hidden; }</style>", unsafe_allow_html=True)
st.title("📘 Advanced AI Research Tool")

# ========== Sidebar Configuration ==========
st.sidebar.header("⚙️ Configuration")

# LLM Provider Selection
provider = st.sidebar.radio(
    "Select LLM Provider",
    ["Cloud (Groq)", "Local (Ollama)"],
    index=0
)

# API Key Handling & LLM Initialization
llm = None
if provider == "Cloud (Groq)":
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.sidebar.error("❌ Groq API Key missing in .env")
    else:
        try:
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                groq_api_key=groq_api_key,
                temperature=0.1  # Very low temp for factual accuracy
            )
            st.sidebar.success("✅ Connected to Groq (Llama 3.1 8B Instant)")
        except Exception as e:
            st.sidebar.error(f"Error connecting to Groq: {e}")

elif provider == "Local (Ollama)":
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        llm = ChatOllama(
            model="llama3.1",
            base_url=ollama_url,
            temperature=0.1
        )
        st.sidebar.success("✅ Connected to Local Ollama")
    except Exception as e:
        st.sidebar.error(f"Error connecting to Ollama: {e}")

# ========== Session State ==========
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_split_docs" not in st.session_state:
    st.session_state.all_split_docs = None
if "url_count" not in st.session_state:
    st.session_state.url_count = 1
if "urls" not in st.session_state:
    st.session_state.urls = [""]

# ========== Layout ==========
left_col, right_col = st.columns([0.85, 2])

# ================ LEFT COLUMN: Data Ingestion ================
with left_col:
    st.subheader("🔗 Add Article URLs")
    temp_urls = []

    for i in range(st.session_state.url_count):
        col_url, col_btn = st.columns([0.9, 0.1])
        with col_url:
            url = st.text_input(
                f"Article URL {i+1}", key=f"url_{i}",
                value=st.session_state.urls[i] if i < len(st.session_state.urls) else ""
            )
            temp_urls.append(url)
        with col_btn:
            if st.button("❌", key=f"remove_{i}"):
                if len(st.session_state.urls) > 1:
                    st.session_state.urls.pop(i)
                    st.session_state.url_count -= 1
                    st.rerun()

    if st.button("➕ Add Another URL"):
        st.session_state.url_count += 1
        st.session_state.urls.append("")
        st.rerun()

    urls = [url for url in temp_urls if url.strip() != ""]

    st.markdown("---")
    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("🔄 Process Research Material"):
        if not urls and not uploaded_files:
            st.warning("Please provide at least one URL or upload a file.")
        else:
            all_documents = []
            status_text = st.empty()

            # Process URLs
            if urls:
                status_text.info(f"📥 Loading {len(urls)} URLs...")
                try:
                    url_loader = UnstructuredURLLoader(urls=urls)
                    url_docs = url_loader.load()
                    for doc in url_docs:
                        doc.metadata["source_type"] = "url"
                    all_documents.extend(url_docs)
                except Exception as e:
                    st.error(f"Error loading URLs: {e}")

            # Process Uploaded Files
            if uploaded_files:
                status_text.info(f"📥 Processing {len(uploaded_files)} files...")
                for uploaded_file in uploaded_files:
                    try:
                        file_ext = uploaded_file.name.split(".")[-1].lower()
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        file_loader = None
                        if file_ext == "pdf":
                            file_loader = PyPDFLoader(uploaded_file.name)
                        elif file_ext == "docx":
                            file_loader = Docx2txtLoader(uploaded_file.name)
                        elif file_ext == "txt":
                            file_loader = TextLoader(uploaded_file.name)

                        if file_loader:
                            file_docs = file_loader.load()
                            for doc in file_docs:
                                doc.metadata["source"] = uploaded_file.name
                                doc.metadata["source_type"] = "file"
                            all_documents.extend(file_docs)

                        os.remove(uploaded_file.name)

                    except Exception as e:
                        st.error(f"Error loading file {uploaded_file.name}: {e}")

            if all_documents:
                # Splitting with optimized parameters
                status_text.info("📚 Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '. ', '? ', '! ', ', ', ' '],
                    chunk_size=600,
                    chunk_overlap=150
                )
                docs = text_splitter.split_documents(all_documents)

                # Embedding with higher quality model
                status_text.info("🧠 Creating embeddings (this may take a moment)...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)

                # Persist
                with open("faiss_store_advanced.pkl", "wb") as f:
                    pickle.dump(vectorstore, f)
                with open("split_docs.pkl", "wb") as f:
                    pickle.dump(docs, f)

                st.session_state.vectorstore = vectorstore
                st.session_state.all_split_docs = docs
                st.session_state.docs_processed = True
                status_text.success(f"✅ Processed & indexed {len(docs)} chunks from {len(all_documents)} documents!")
            else:
                status_text.error("No content could be extracted.")

# ================ RIGHT COLUMN: Q&A ================
with right_col:
    st.subheader("❓ Ask Your Research Question")

    # Load vectorstore if not in memory but file exists
    if not st.session_state.vectorstore and os.path.exists("faiss_store_advanced.pkl"):
        with open("faiss_store_advanced.pkl", "rb") as f:
            st.session_state.vectorstore = pickle.load(f)
        if os.path.exists("split_docs.pkl"):
            with open("split_docs.pkl", "rb") as f:
                st.session_state.all_split_docs = pickle.load(f)
        st.session_state.docs_processed = True

    query = st.text_input("Enter your question", key="q_input")

    if st.button("Submit Question"):
        if not query:
            st.warning("Please enter a question.")
        elif not st.session_state.docs_processed:
            st.warning("Please process research material first.")
        else:
            # ====== LAYER 1: Vague Question Guard (Regex) ======
            if is_vague_query(query):
                st.warning(
                    "🚫 **This is not a research question.**\n\n"
                    "I am a specialized research assistant that **only answers questions "
                    "based on the documents you have provided**.\n\n"
                    "Please ask a specific question related to the content of your uploaded articles or documents."
                )

            # ====== LAYER 1.5: LLM Intent Classification ======
            elif llm is None:
                st.error("LLM not initialized. Check your settings/API keys in the sidebar.")
            else:
                intent_prompt = f"""Classify the intent of this question. Answer with ONLY one word: RESEARCH, MIXED, or NONSENSE.

- RESEARCH: A genuine question asking for factual, technical, or analytical information about a topic. Example: "What is MakerDAO?"
- MIXED: A question that combines a nonsensical/absurd premise WITH a valid research question. Example: "how would you marry a blockchain agent and how does blockchain security work?"
- NONSENSE: A question that is entirely silly, irrelevant, inappropriate, personal, or makes no logical sense as a research query. Example: "how would you marry a blockchain agent?"

Question: {query}

Answer (RESEARCH, MIXED, or NONSENSE only):"""

                with st.spinner("Analyzing question intent..."):
                    try:
                        intent_response = llm.invoke(intent_prompt)
                        intent = intent_response.content.strip().upper().split()[0]  # get first word
                    except Exception:
                        intent = "RESEARCH"  # fail-safe: proceed if intent check fails

                if intent == "NONSENSE":
                    st.warning(
                        "🚫 **This is not a valid research question.**\n\n"
                        "Your question doesn't appear to be asking for factual or research-based information.\n\n"
                        "Please ask a clear, specific question about the topics covered in your documents."
                    )

                else:
                    vectorstore = st.session_state.vectorstore

                    # ====== LAYER 2: Relevance Gate (FAISS Score Check) ======
                    with st.spinner("🔍 Searching documents for relevant information..."):
                        relevant_docs, is_relevant = check_relevance_with_scores(
                            vectorstore, query, k=8, threshold=1.8
                        )

                    if not is_relevant:
                        st.warning(
                            "🔍 **No relevant information found.**\n\n"
                            "I searched through all your provided documents but could not find "
                            "content related to your question.\n\n"
                            "This means the answer to your question is **not present** in the documents you uploaded. "
                            "Please ask a question that relates to the content of your articles/documents."
                        )
                    else:
                        # ====== LAYER 3: Hybrid Retrieval (BM25 + FAISS) ======
                        try:
                            # Semantic retriever (FAISS)
                            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                            # Keyword retriever (BM25)
                            all_split_docs = st.session_state.all_split_docs
                            if all_split_docs:
                                bm25_retriever = BM25Retriever.from_documents(all_split_docs)
                                bm25_retriever.k = 4

                                # Ensemble: 60% semantic + 40% keyword
                                hybrid_retriever = EnsembleRetriever(
                                    retrievers=[faiss_retriever, bm25_retriever],
                                    weights=[0.6, 0.4]
                                )
                            else:
                                hybrid_retriever = faiss_retriever

                            # ====== LAYER 4: Anti-Hallucination Prompt ======
                            custom_prompt = PromptTemplate(
                                input_variables=["context", "question"],
                                template="""You are a research assistant. Your job is to provide thorough, detailed answers using the provided context.

RULES:
1. Base your answer ONLY on the information in the context below.
2. You MAY synthesize technical concepts, but DO NOT infer answers for nonsensical, biological, or metaphorical questions (e.g., "marrying" an agent, "killing" a process) unless strictly defined in the text.
3. If a question has multiple parts, and one part is nonsensical or not in the context, explicitly state "The documents do not mention [concept]" for that part, and answer ONLY the valid technical parts.
4. If the context does NOT contain information to answer the question, respond EXACTLY with: "The provided documents do not contain information about this topic."
5. NEVER add facts, statistics, course names, university names, or any specific information that is NOT in the context.
6. Cite your sources using [Source: filename or URL] for each key point.
7. Use clear formatting: headings, bullet points, bold for key terms.

Context:
{context}

Question: {question}

Detailed Answer:"""
                            )

                            chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                retriever=hybrid_retriever,
                                chain_type="stuff",
                                chain_type_kwargs={"prompt": custom_prompt},
                                return_source_documents=True
                            )

                            with st.spinner("🧠 Generating answer from documents..."):
                                result = chain.invoke({"query": query})
                                answer = result["result"]
                                source_docs = result["source_documents"]

                            # ====== LAYER 5: Answer Post-Processing ======
                            # Check if the LLM itself said "no information found"
                            no_info_phrases = [
                                "do not contain information",
                                "does not contain information",
                                "couldn't find information",
                                "could not find information",
                                "no relevant information",
                                "not mentioned in the provided",
                                "not found in the provided",
                                "not present in the provided",
                                "not available in the provided",
                                "i don't have enough context",
                                "the context does not",
                                "there is no information",
                            ]

                            answer_lower = answer.lower()
                            is_no_info = any(phrase in answer_lower for phrase in no_info_phrases)

                            if is_no_info:
                                st.warning(
                                    "🔍 **No relevant information found in your documents.**\n\n"
                                    "The AI analyzed the retrieved content but determined that your documents "
                                    "do not contain information to answer this question.\n\n"
                                    "Please ask a question directly related to the content of your uploaded articles."
                                )
                            else:
                                # Display Answer
                                st.markdown("### 📝 Research Analysis")
                                st.markdown(answer)

                                # Display Citations
                                st.markdown("---")
                                st.subheader("📚 Sources Referenced")

                                unique_sources = {}
                                for doc in source_docs:
                                    source_name = doc.metadata.get("source", "Unknown")
                                    if source_name not in unique_sources:
                                        snippet = doc.page_content[:250].replace("\n", " ").strip()
                                        unique_sources[source_name] = snippet

                                for source_name, snippet in unique_sources.items():
                                    source_type = "🔗" if source_name.startswith("http") else "📄"
                                    if source_name.startswith("http"):
                                        st.markdown(f"{source_type} [{source_name}]({source_name})")
                                    else:
                                        st.markdown(f"{source_type} **{source_name}**")
                                    st.caption(f'"{snippet}..."')

                        except Exception as e:
                            st.error(f"❌ Error generating answer: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("ResearchRAG-AI | Hybrid Search | Anti-Hallucination")
