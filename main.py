import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import hashlib

import os
from dotenv import load_dotenv
load_dotenv()

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# chat storage
chat_history = []
# Step1 :load the document

PDF_PATH = "input/notes.pdf"
HASH_PATH = "chroma_db_source.sha256"

with open(PDF_PATH, "rb") as pdf_file:
    current_hash = hashlib.sha256(pdf_file.read()).hexdigest()

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()


for doc in documents:
    doc.page_content = re.sub(
        r'\s+',
        ' ',
        doc.page_content
    ).strip()

# step2 : split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)



# step3 : create embeddings for the chunks
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



DB_PATH = "chroma_db"

saved_hash = None

if os.path.exists(HASH_PATH):
    with open(HASH_PATH, "r") as hash_file:
        saved_hash = hash_file.read().strip()

if os.path.isdir(DB_PATH) and saved_hash == current_hash:
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    print("Existing vector store loaded!")


else:
    # Remove the old database if it exists
    if os.path.isdir(DB_PATH):
        shutil.rmtree(DB_PATH)

    # Create a new vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    # Save the PDF fingerprint
    with open(HASH_PATH, "w") as hash_file:
        hash_file.write(current_hash)

    print("Vector store created or updated!")

   
# step5 : retrieve the vector store from the persisted directory

retriever = vectorstore.as_retriever(
     search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)



# step 6 : create a prompt template for the question answering



prompt = ChatPromptTemplate.from_template(
    """
    You are a reliable assistant that answers questions
    using the provided PDF document.

    Chat history:
    {chat_history}

    Context from the PDF:
    {context}

    Current question:
    {question}

    Follow these rules strictly:

    1. Use the PDF context as the only source of factual
       information for questions about the document.

    2. Before answering, check whether the context actually
       supports the requested information.

    3. If the context does not contain enough information,
       respond:
       "I could not find sufficient information in the document."

    4. If the context supports only part of the answer,
       explain what the document establishes and what is missing.

    5. Never invent facts, figures, examples, quotations,
       or details that are not supported by the context.

    6. Answer the exact question. Avoid unrelated explanations
       and unnecessary repetition.

    7. For multi-part questions, organize the answer using
       numbered points or bullet points where appropriate.

    8. Use chat history to understand follow-up questions,
       but do not treat previous assistant responses as
       evidence that a claim is present in the PDF.

    9. Respond naturally to greetings s thanks.

    10. Keep the answer clear, direct, and appropriately detailed.
    """
)


# step 7 : create the model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=1000
)



contextualize_prompt = ChatPromptTemplate.from_template(
    """
    Given the chat history and the latest user question,
    rewrite the latest question so that it can be understood
    independently.

    Do NOT answer the question.

    Chat history:
    {chat_history}

    Latest question:
    {question}

    Standalone question:
    """
)

contextualize_chain = (
    contextualize_prompt
    | model
    | StrOutputParser()
)



def conversational_rag(question):

    # Step 1: Format conversation history
    recent_history = chat_history[-10:]  
    history_text = "\n".join(
        f"{role}: {content}"
        for role, content in recent_history
    )

    # Step 2: Rewrite the question using conversation history
    standalone_question = contextualize_chain.invoke({
        "chat_history": history_text,
        "question": question
    }).strip()

    # Step 3: Retrieve relevant PDF chunks
    docs = retriever.invoke(standalone_question)

    # Step 4: Combine chunks into context
    context = format_docs(docs)

    # Step 5: Generate the answer
    answer = (
        prompt
        | model
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": question,
        "chat_history": history_text
    })

    # Step 6: Save conversation history
    chat_history.append(("human", question))
    chat_history.append(("ai", answer))

    return answer, docs


# Step 9: Interactive chat loop
while True:
    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not question:
        continue

    answer, docs = conversational_rag(question)

    print("\nAssistant:", answer)
   