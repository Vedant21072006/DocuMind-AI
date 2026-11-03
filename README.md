# 📚 DocuMind AI — Chat with Your Documents

**An AI-powered PDF assistant that lets you ask questions about your documents and receive contextual answers with page references.**

🔗 **Live Demo:** [Try DocuMind AI](https://documind-ai-ndvgsx3ztzw2h5ngymqsuz.streamlit.app/)

## ✨ Overview

DocuMind AI transforms static PDF documents into interactive knowledge sources. Upload a PDF, ask questions in natural language, and get answers grounded in the document's content.

Built using Retrieval-Augmented Generation (RAG), the application retrieves relevant document passages before generating responses, helping users explore information without manually searching through pages.

## 🚀 Features

- 📄 **PDF Upload:** Upload a PDF document and process its content.
- 💬 **Conversational Q&A:** Ask questions using natural language.
- 🔍 **Semantic Search:** Retrieve relevant passages using vector embeddings.
- 🧠 **Retrieval-Augmented Generation:** Generate context-aware responses grounded in retrieved document content.
- 📌 **Page References:** View source-page references alongside answers.
- 🔄 **Conversation Context:** Ask follow-up questions with conversational history.
- 🎯 **MMR Retrieval:** Retrieve diverse, relevant chunks to improve contextual coverage.
- 🖥️ **Interactive UI:** A modern, chat-style interface built with Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Web interface |
| LangChain | LLM orchestration and document processing |
| Hugging Face Sentence Transformers | Text embeddings |
| ChromaDB | Vector storage and similarity search |
| Groq API | Large language model inference |
| PyPDF | PDF text extraction |

## ⚙️ How It Works

1. **Upload:** The user uploads a PDF document.
2. **Extract:** Text is extracted from the PDF.
3. **Chunk:** The document is split into smaller, overlapping text chunks.
4. **Embed:** Text chunks are converted into vector embeddings.
5. **Index:** Embeddings are stored in ChromaDB.
6. **Retrieve:** A question is used to retrieve relevant chunks through Maximum Marginal Relevance (MMR).
7. **Generate:** The language model uses the retrieved context to formulate an answer.
8. **Reference:** The application displays relevant source-page references.

## 🏗️ Architecture

```text
                ┌──────────────────┐
                │    User / PDF    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │   Streamlit UI   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  PDF Extraction  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Text Chunking    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Hugging Face     │
                │ Embeddings       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    ChromaDB      │
                └────────┬─────────┘
                         │
              User Question
                         │
                         ▼
                ┌──────────────────┐
                │ Relevant Context │
                │ Retrieval (MMR)  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Groq LLM      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Answer + Sources │
                └──────────────────┘
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vedant21072006/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Obtain your API key from [Groq Console](https://console.groq.com/).

**Security:** Never commit your `.env` file or expose your API key in source code.

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🔐 Security and Limitations

- API credentials should be stored in environment variables or deployment secrets.
- Responses depend on the information available in the uploaded document and the retrieved passages.
- The application may occasionally generate incorrect answers; verify important information against the original PDF.
- Processing time depends on document size, embedding generation, and model response latency.
- The deployed version depends on the availability and usage limits of its external model API.

## 🗺️ Future Improvements

- [ ] Support multiple PDFs in a single knowledge base.
- [ ] Add persistent vector storage and document management.
- [ ] Improve retrieval with reranking and hybrid search.
- [ ] Add streaming responses and richer source previews.
- [ ] Support additional document formats.
- [ ] Add automated evaluation for retrieval quality and answer faithfulness.
- [ ] Improve deployment monitoring, error handling, and performance.

## 🎯 Learning Outcomes

This project explores practical LLM application engineering, including:

- Building a Retrieval-Augmented Generation pipeline.
- Extracting and chunking unstructured PDF data.
- Generating embeddings and performing semantic retrieval.
- Using LangChain to orchestrate LLM workflows.
- Integrating hosted language models through an API.
- Managing secrets and deploying an interactive AI application.

## 👨‍💻 Author

**Vedant Rathod**

B.Tech Computer Science Engineering | Applied AI & LLM Engineering

- GitHub: [@Vedant21072006](https://github.com/Vedant21072006)
- LeetCode: [Vedant021](https://leetcode.com/u/Vedant021/)

## 📄 License

This project is intended for educational and portfolio purposes. Add a `LICENSE` file if you wish to distribute it under a specific open-source license.

---

⭐ If you find this project interesting, consider starring the repository!
