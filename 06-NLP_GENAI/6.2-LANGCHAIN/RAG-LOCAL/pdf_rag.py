import os

import ollama
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

##INIT
pdf_file = input("Enter the File Path:")
if len(pdf_file) <= 0:
    print("Appending Default PDF...")
    pdf_file = "./data/UPDATED_CV.pdf"

embedding_model_name = "nomic-embed-text:latest"
llm_model_name = "llama3.2:latest"
documents = ""

##LOADING PDF
try:
    pdf_loader = UnstructuredLoader(file_path=pdf_file)
    documents = pdf_loader.load()
    print("Documents Loaded Successfully")
except Exception as E:
    print(E)

##CHUNKING
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
doc_chunks = splitter.split_documents(documents)
for doc in doc_chunks:
    doc.metadata = {}
print(f"Document splitted into {len(doc_chunks)}.")

##EMBEDDING CHUNKS TO VECTOR DB
ollama.pull(embedding_model_name)
vector_store = Chroma.from_documents(
    documents=doc_chunks,
    embedding=OllamaEmbeddings(model=embedding_model_name),
    collection_name="PERSONAL_RESUME",
)

llm = ChatOllama(model=llm_model_name)
multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an AI assistant. Generate 5 different versions of the user question "
        "to improve document retrieval from a vector database.\n"
        "Original question: {question}"
    ),
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm,
    prompt=multi_query_prompt,
)

context_prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below:\n{context}\n\nQuestion: {question}"
)


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | context_prompt
    | llm
    | StrOutputParser()
)

user_question = input("What do you want to check in Resume?:\n")
print("\nAssistant: ", end="", flush=True)

# Iterate through the chunks as they arrive
for chunk in rag_chain.stream(user_question):
    print(chunk, end="", flush=True)
print()
