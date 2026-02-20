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
            # ====== Pre-processing: Strip gaming prefixes ======
            # Removes phrases like "as per the document I provided" that users
            # add to try to trick the system into answering off-topic questions.
            GAMING_PREFIXES = [
                r"^as per (the |my )?(documents?|articles?|files?|sources?)\s*(i\s*(have\s*)?)?(provided|uploaded|shared|given)[,.]?\s*",
                r"^based on (the |my )?(documents?|articles?|files?|sources?)\s*(i\s*(have\s*)?)?(provided|uploaded|shared|given)[,.]?\s*",
                r"^from (the |my )?(documents?|articles?|files?|sources?)\s*(i\s*(have\s*)?)?(provided|uploaded|shared|given)[,.]?\s*",
                r"^according to (the |my )?(documents?|articles?|files?|sources?)[,.]?\s*",
                r"^using (the |my )?(documents?|articles?|files?|sources?)[,.]?\s*",
                r"^in (the |my )?(documents?|articles?|files?|sources?)[,.]?\s*",
            ]
            cleaned_query = query.strip()
            for prefix_pattern in GAMING_PREFIXES:
                cleaned_query = re.sub(prefix_pattern, "", cleaned_query, flags=re.IGNORECASE).strip()

            # ====== LAYER 1: Vague Question Guard (Regex) ======
            if is_vague_query(cleaned_query):
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

- RESEARCH: A genuine question asking for factual, technical, or analytical information about a topic.
- MIXED: Contains one nonsensical/absurd part AND one valid research part.
- NONSENSE: Entirely silly, irrelevant, inappropriate, personal, or makes no logical sense as a research query.

Question: {cleaned_query}

Answer (RESEARCH, MIXED, or NONSENSE only):"""

                with st.spinner("Analyzing question intent..."):
                    try:
                        intent_response = llm.invoke(intent_prompt)
                        intent = intent_response.content.strip().upper().split()[0]
                    except Exception:
                        intent = "RESEARCH"

                if intent == "NONSENSE":
                    st.warning(
                        "🚫 **This is not a valid research question.**\n\n"
                        "Your question doesn't appear to be asking for factual or research-based information.\n\n"
                        "Please ask a clear, specific question about the topics covered in your documents."
                    )

                else:
                    vectorstore = st.session_state.vectorstore

                    # ====== LAYER 2: FAISS Relevance Gate ======
                    with st.spinner("🔍 Searching documents for relevant information..."):
                        relevant_docs, is_relevant = check_relevance_with_scores(
                            vectorstore, cleaned_query, k=8, threshold=1.8
                        )

                    if not is_relevant:
                        st.warning(
                            "🔍 **No relevant information found.**\n\n"
                            "I searched through all your provided documents but could not find "
                            "content related to your question.\n\n"
                            "Please ask a question that relates to the content of your articles/documents."
                        )
                    else:
                        try:
                            # ====== LAYER 3: Targeted Hybrid Retrieval ======
                            # Use MMR (Maximal Marginal Relevance) for diversity + precision
                            # fetch_k=20 candidates, return only top k=3 diverse, non-redundant chunks
                            # This drastically reduces token usage vs fetching 8 similar chunks
                            faiss_retriever = vectorstore.as_retriever(
                                search_type="mmr",
                                search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.7}
                            )

                            all_split_docs = st.session_state.all_split_docs
                            if all_split_docs:
                                bm25_retriever = BM25Retriever.from_documents(all_split_docs)
                                bm25_retriever.k = 3
                                hybrid_retriever = EnsembleRetriever(
                                    retrievers=[faiss_retriever, bm25_retriever],
                                    weights=[0.65, 0.35]
                                )
                            else:
                                hybrid_retriever = faiss_retriever

                            # Retrieve candidate chunks
                            candidate_chunks = hybrid_retriever.invoke(cleaned_query)

                            # ====== LAYER 2.5: Context Sufficiency Check ======
                            # Core anti-hallucination gate: asks LLM if retrieved context
                            # actually ANSWERS the question — not just mentions related keywords.
                            context_text = "\n\n---\n\n".join([doc.page_content for doc in candidate_chunks])

                            sufficiency_prompt = f"""You are a strict relevance judge.

Your task: Determine if the CONTEXT below directly contains the information needed to answer the QUESTION.

QUESTION: {cleaned_query}

CONTEXT:
{context_text}

Rules:
- Answer YES only if the context has specific, direct information that answers the question.
- Answer NO if the context only loosely mentions related keywords but doesn't actually address the question.
- Answer NO if answering the question would require knowledge NOT present in the context.
- Example: If question asks "how to build blockchain architecture for a project" and context is about "blockchain courses and trading platforms", answer NO.
- Example: If question asks "what is MakerDAO" and context explains MakerDAO's rules and tokens, answer YES.

Answer with ONLY one word: YES or NO"""

                            with st.spinner("🔎 Verifying if documents contain the answer..."):
                                try:
                                    sufficiency_response = llm.invoke(sufficiency_prompt)
                                    is_sufficient = sufficiency_response.content.strip().upper().startswith("YES")
                                except Exception:
                                    is_sufficient = True  # fail-safe

                            if not is_sufficient:
                                st.warning(
                                    "📭 **The documents do not contain an answer to this question.**\n\n"
                                    "Your documents mention related topics, but do not contain specific "
                                    "information to answer this particular question.\n\n"
                                    "Please ask something that is **directly covered** in your articles/documents."
                                )
                            else:
                                # ====== LAYER 4: Precise Answer Generation ======
                                custom_prompt = PromptTemplate(
                                    input_variables=["context", "question"],
                                    template="""You are a research assistant. Answer the question using ONLY the information in the context below.

STRICT RULES:
1. Use ONLY information explicitly present in the context. Zero external knowledge.
2. If any part of the question cannot be answered from the context, skip that part silently.
3. Write a detailed, well-structured answer with headings and bullet points.
4. Do NOT include inline source citations (e.g., [Source: ...]) IN the answer body — sources will be listed separately at the end.
5. End your answer with a section called "## 📌 Key Topics Used" that lists 2-5 specific topics/sections from the context that you used to answer.

Context:
{context}

Question: {question}

Answer:"""
                                )

                                chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    retriever=hybrid_retriever,
                                    chain_type="stuff",
                                    chain_type_kwargs={"prompt": custom_prompt},
                                    return_source_documents=True
                                )

                                with st.spinner("🧠 Generating precise answer from documents..."):
                                    result = chain.invoke({"query": cleaned_query})
                                    answer = result["result"]
                                    source_docs = result["source_documents"]

                                # ====== LAYER 5: Post-Processing — no-info leak check ======
                                no_info_phrases = [
                                    "do not contain information",
                                    "does not contain information",
                                    "couldn't find information",
                                    "could not find information",
                                    "no relevant information",
                                    "not mentioned in the provided",
                                    "not found in the provided",
                                    "the context does not",
                                    "there is no information",
                                    "cannot be answered",
                                    "is not discussed",
                                    "is not covered",
                                ]
                                is_no_info = any(p in answer.lower() for p in no_info_phrases)

                                if is_no_info:
                                    st.warning(
                                        "📭 **No relevant information found in your documents.**\n\n"
                                        "The AI confirmed your documents do not contain the answer. "
                                        "Please ask something directly covered in your articles."
                                    )
                                else:
                                    # ====== Display Answer ======
                                    st.markdown("### 📝 Research Analysis")
                                    st.markdown(answer)

                                    # ====== Display Sources at the End ======
                                    st.markdown("---")
                                    st.markdown("### 🗂️ Sources Used")

                                    # Collect unique sources
                                    seen_sources = {}
                                    for doc in source_docs:
                                        src = doc.metadata.get("source", "Unknown")
                                        if src not in seen_sources:
                                            seen_sources[src] = doc.page_content[:200].replace("\n", " ").strip()

                                    for i, (src, snippet) in enumerate(seen_sources.items(), 1):
                                        if src.startswith("http"):
                                            st.markdown(f"**{i}.** 🔗 [{src}]({src})")
                                        else:
                                            st.markdown(f"**{i}.** 📄 `{src}`")

                        except Exception as e:
                            st.error(f"❌ Error generating answer: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("ResearchRAG-AI | Hybrid Search | Anti-Hallucination")
