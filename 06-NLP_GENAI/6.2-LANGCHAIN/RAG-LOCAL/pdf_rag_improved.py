import os
import sys
from typing import List

# Use specialized packages (ensure you have these installed)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

# Configuration Constants
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2"
COLLECTION_NAME = "personal_resume"
PERSIST_DIRECTORY = "./chroma_db"


def load_and_split_pdf(file_path: str) -> List[Document]:
    """Loads a PDF and splits it into manageable chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at {file_path}")

    loader = UnstructuredLoader(file_path=file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Standard practice: Clear complex metadata that causes vector DB issues
    for chunk in chunks:
        chunk.metadata = {}

    return chunks


def initialize_rag_chain(chunks: List[Document]):
    """Sets up the vector store and the RAG chain."""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=LLM_MODEL)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context:\n{context}\n\n"
        "Question: {question}"
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    pdf_path = input("Enter PDF path (default ./data/UPDATED_CV.pdf): ").strip()
    if not pdf_path:
        pdf_path = "./data/UPDATED_CV.pdf"

    try:
        print("Processing document...")
        chunks = load_and_split_pdf(pdf_path)
        chain = initialize_rag_chain(chunks)

        user_question = input("\nWhat would you like to ask your resume? ")

        print("\nAssistant: ", end="", flush=True)
        for chunk in chain.stream(user_question):
            print(chunk, end="", flush=True)
        print()

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
