
import os
import re
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
}

.stApp {
    background: #0b0d12;
    color: #edf0f7;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(11, 13, 18, 0.95);
}

[data-testid="stSidebar"] {
    background: #10131b;
    border-right: 1px solid #242834;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.6rem;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.7px;
}

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    letter-spacing: -1px;
    color: #f3f4f8;
}

.brand span {
    color: #9c8cff;
}

.brand-subtitle {
    color: #9299aa;
    font-size: 12px;
    margin-top: 4px;
}

.sidebar-label {
    color: #858da0;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 1.7px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 10px;
}

.status-pill {
    display: inline-block;
    border: 1px solid #303648;
    background: #181c27;
    border-radius: 30px;
    padding: 7px 11px;
    color: #c6cadd;
    font-size: 11px;
}

.hero {
    padding: 65px 12px 25px 12px;
    text-align: center;
}

.hero-icon {
    width: 66px;
    height: 66px;
    margin: 0 auto 25px auto;
    border: 1px solid #3a345e;
    border-radius: 22px;
    background: linear-gradient(135deg, #24203e, #151a2a);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 29px;
    box-shadow: 0 0 55px rgba(139, 115, 255, 0.09);
}

.hero h1 {
    font-size: clamp(32px, 5vw, 48px);
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 15px;
}

.gradient-text {
    background: linear-gradient(90deg, #b5a5ff, #8f9dff, #8bd8f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #9aa1b2;
    font-size: 15px;
    max-width: 490px;
    line-height: 1.8;
    margin: 0 auto;
}

.feature-card {
    height: 100%;
    min-height: 130px;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #272c3a;
    background: linear-gradient(145deg, #151925, #11141c);
}

.feature-icon {
    font-size: 21px;
    margin-bottom: 14px;
}

.feature-title {
    color: #f0f1f8;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 6px;
}

.feature-desc {
    color: #939bad;
    font-size: 12px;
    line-height: 1.6;
}

.section-caption {
    color: #9098aa;
    font-size: 12px;
    margin: 30px 0 12px 0;
}

div[data-testid="stChatMessage"] {
    border: 1px solid #272c39;
    background: #121620;
    border-radius: 16px;
    padding: 16px 19px;
    margin-bottom: 15px;
}

div[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background: #171a27;
}

.stChatInput textarea {
    background: #151924 !important;
    color: #f3f4fa !important;
}

div[data-testid="stChatInput"] {
    border-color: #353b4d !important;
}

.stButton > button {
    border: 1px solid #303648;
    background: #171b26;
    color: #e9eaf3;
    border-radius: 12px;
    min-height: 43px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #8e7cff;
    color: white;
    background: #211e36;
}

button[kind="primary"] {
    background: #8170ed !important;
    color: white !important;
    border: none !important;
}

button[kind="primary"]:hover {
    background: #9284fa !important;
}

div[data-testid="stFileUploader"] {
    background: #151924;
    border: 1px dashed #444b61;
    border-radius: 14px;
    padding: 8px;
}

div[data-testid="stExpander"] {
    border: 1px solid #2b3040;
    border-radius: 12px;
    background: #121620;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #292e3b;
}

#MainMenu, footer {
    visibility: hidden;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 6rem 1rem;
    }

    .hero {
        padding-top: 35px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# AI COMPONENTS
# --------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_model():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        max_tokens=1000
    )


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


answer_prompt = ChatPromptTemplate.from_template("""
You are a reliable assistant that answers questions using
the provided PDF document.

Chat history:
{chat_history}

Context from the PDF:
{context}

Current question:
{question}

Follow these rules:
1. Use the PDF context as the factual source for document questions.
2. If the context does not support an answer, say:
   "I could not find sufficient information in the document."
3. If only part of the answer is supported, explain what is
   available and what is missing.
4. Never invent facts, figures, quotations, or examples.
5. Answer the exact question clearly and directly.
6. Structure multi-part answers with bullets when helpful.
7. Use history to understand follow-up questions, but do not
   treat previous assistant answers as proof of document facts.
8. Respond naturally to greetings and thanks.
9. Keep answers appropriately detailed.
""")

contextualize_prompt = ChatPromptTemplate.from_template("""
Rewrite the latest user question as a standalone question that
can be understood without the previous conversation.

Use the chat history only to resolve references such as "it",
"they", "that", or "what about its advantages".

If the question is already understandable on its own, preserve
its meaning without adding assumptions.

Do not answer the question.

Chat history:
{chat_history}

Latest question:
{question}

Standalone question:
""")


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def process_pdf(pdf_bytes, filename):
    temp_pdf = None
    db_directory = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_pdf = temp_file.name

        loader = PyPDFLoader(temp_pdf)
        documents = loader.load()

        for doc in documents:
            doc.page_content = re.sub(
                r"\s+", " ", doc.page_content
            ).strip()

        documents = [
            doc for doc in documents
            if doc.page_content
        ]

        if not documents:
            raise ValueError(
                "No readable text was found in this PDF."
            )

        for doc in documents:
            doc.metadata["source"] = filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "Could not create searchable text chunks from this PDF."
            )

        db_directory = tempfile.mkdtemp(prefix="documind_chroma_")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=db_directory
        )

        return vectorstore, db_directory, len(documents)

    finally:
        if temp_pdf and os.path.exists(temp_pdf):
            os.remove(temp_pdf)


# --------------------------------------------------
# CONVERSATIONAL RAG
# --------------------------------------------------

def answer_question(question):
    history = st.session_state.chat_history[-10:]

    history_text = "\n".join(
        f"{role}: {content}"
        for role, content in history
    )

    model = get_model()

    standalone_question = (
        contextualize_prompt
        | model
        | StrOutputParser()
    ).invoke({
        "chat_history": history_text,
        "question": question
    }).strip()

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    answer = (
        answer_prompt
        | model
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": standalone_question,
        "chat_history": history_text
    })

    st.session_state.chat_history.append(
        ("human", question)
    )
    st.session_state.chat_history.append(
        ("ai", answer)
    )

    return answer, docs


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "vectorstore": None,
    "db_directory": None,
    "processed_hash": None,
    "document_name": None,
    "page_count": 0,
    "chat_history": [],
    "messages": [],
    "pending_question": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="brand">DocuMind<span>.</span></div>
    <div class="brand-subtitle">Your documents, understood.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Workspace</div>',
                unsafe_allow_html=True)

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown('<div class="sidebar-label">Your document</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload one PDF to start asking questions."
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if file_hash != st.session_state.processed_hash:
            try:
                with st.spinner(
                    "Reading your PDF and building its search index..."
                ):
                    vectorstore, db_directory, page_count = process_pdf(
                        pdf_bytes,
                        uploaded_file.name
                    )

                st.session_state.vectorstore = vectorstore
                st.session_state.db_directory = db_directory
                st.session_state.processed_hash = file_hash
                st.session_state.document_name = uploaded_file.name
                st.session_state.page_count = page_count
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.session_state.pending_question = None

                st.success("Your document is ready!")

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")

                st.session_state.vectorstore = None
                st.session_state.processed_hash = None

        if (
            st.session_state.processed_hash == file_hash
            and st.session_state.vectorstore is not None
        ):
            st.markdown(
                f"""
                <div style="
                    background:#181d29;
                    border:1px solid #343a4d;
                    padding:13px;
                    border-radius:12px;
                    margin-top:8px;
                ">
                    <div style="font-size:22px;">📄</div>
                    <div style="
                        font-size:13px;
                        font-weight:600;
                        overflow-wrap:anywhere;
                        margin-top:8px;
                    ">{uploaded_file.name}</div>
                    <div style="
                        color:#959db0;
                        font-size:11px;
                        margin-top:5px;
                    ">{st.session_state.page_count} pages with readable text</div>
                    <div style="
                        color:#8fddbd;
                        font-size:11px;
                        margin-top:8px;
                    ">● Indexed and ready</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore = None
            st.session_state.processed_hash = None
            st.session_state.document_name = None
            st.session_state.page_count = 0
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.session_state.pending_question = None

        st.caption("Upload a PDF to activate your workspace.")

    st.markdown("---")

    st.markdown('<div class="sidebar-label">Capabilities</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:13px;line-height:2.3;color:#a8afbf;">
        ✦  Context-aware answers<br>
        ✦  Follow-up conversations<br>
        ✦  Page-level source references<br>
        ✦  Private document workspace
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="status-pill">✦ Powered by LangChain + Groq</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

left, right = st.columns([0.75, 0.25])

with left:
    st.markdown("### AI Document Assistant")

with right:
    if st.session_state.vectorstore is not None:
        st.markdown(
            '<div style="text-align:right;padding-top:10px;'
            'color:#8fddbd;font-size:12px;">● Document connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align:right;padding-top:10px;'
            'color:#9299aa;font-size:12px;">○ Awaiting document</div>',
            unsafe_allow_html=True
        )

st.markdown("---")


# --------------------------------------------------
# WELCOME SCREEN
# --------------------------------------------------

if st.session_state.vectorstore is None:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">✧</div>
        <h1>Chat with your <span class="gradient-text">documents.</span></h1>
        <p>
            Turn lengthy PDFs into clear answers.
            Ask questions, explore ideas, and find the information
            that matters — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-caption" '
                'style="text-align:center;">WHAT YOU CAN DO</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⌕</div>
            <div class="feature-title">Ask anything</div>
            <div class="feature-desc">
                Find explanations and details hidden inside your PDF.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">◈</div>
            <div class="feature-title">Understand more</div>
            <div class="feature-desc">
                Ask follow-up questions without losing context.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">↗</div>
            <div class="feature-title">Check sources</div>
            <div class="feature-desc">
                See the PDF pages used to generate an answer.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        text-align:center;
        color:#828a9e;
        font-size:12px;
        margin-top:30px;
    ">
        Get started by uploading a PDF in the left sidebar.
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander(
                    f"📚 Sources · {len(message['sources'])} page reference(s)"
                ):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['filename']}** · Page {source['page']}"
                        )


# --------------------------------------------------
# SUGGESTED QUESTIONS
# --------------------------------------------------

if (
    st.session_state.vectorstore is not None
    and not st.session_state.messages
):
    st.markdown(
        '<div class="section-caption">TRY ASKING</div>',
        unsafe_allow_html=True
    )

    suggestions = [
        "Summarize this document",
        "What are the key takeaways?",
        "Explain the most important concepts"
    ]

    cols = st.columns(3)

    for index, suggestion in enumerate(suggestions):
        with cols[index]:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True
            ):
                st.session_state.pending_question = suggestion
                st.rerun()


# --------------------------------------------------
# CHAT INPUT AND ANSWERS
# --------------------------------------------------

typed_question = st.chat_input(
    "Ask anything about your document...",
    disabled=st.session_state.vectorstore is None
)

question = typed_question

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question and st.session_state.vectorstore is not None:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking through your document..."):
                answer, docs = answer_question(question)

            st.markdown(answer)

            unique_sources = {}
            for doc in docs:
                page_index = doc.metadata.get("page")
                filename = doc.metadata.get(
                    "source",
                    st.session_state.document_name or "Uploaded PDF"
                )

                if page_index is not None:
                    unique_sources[(filename, page_index + 1)] = True

            sources = [
                {"filename": filename, "page": page}
                for filename, page in unique_sources
            ]

            if sources:
                with st.expander(
                    f"📚 Sources · {len(sources)} page reference(s)"
                ):
                    for source in sources:
                        st.markdown(
                            f"**{source['filename']}** · Page {source['page']}"
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    except Exception as exc:
        error_message = (
            "Something went wrong while generating the answer. "
            "Please try again."
        )
        st.error(error_message)
        st.caption(str(exc))

    st.rerun()
