from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import textwrap


def get_pdf_pages(pdf_file):
    loader = PyPDFLoader("paper.pdf")
    pages = loader.load_and_split()

    return pages

def initialize_embeddings():
    embeddings = OllamaEmbeddings()
    return embeddings

def initialize_llm():
    llm = Ollama(model="llama2")
    return llm

def get_chroma_vectors_db(pages):

    embeddings = initialize_embeddings()
    vector_db = Chroma.from_documents(pages, embeddings)

    return vector_db

def get_question_answer_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                return_source_documents=False)

    return qa_chain

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    return(wrap_text_preserve_newlines(llm_response['result']))
