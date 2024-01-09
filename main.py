import argparse
from utils import get_pdf_pages, get_chroma_vectors_db, \
                get_question_answer_chain, process_llm_response, initialize_llm

# parse arguments
parser = argparse.ArgumentParser(description='PDF chatbot using llama2 LLM')
parser.add_argument('--pdf_file', type=str, help='path to pdf file', default='pdfs/paper.pdf')
parser.add_argument('--model', type=str, default='llama2')
args = parser.parse_args()

#parse arguments
pdf_file = args.pdf_file
model = args.model

# get pages from pdf file
pages = get_pdf_pages(pdf_file)

# get chroma vectors db
vector_db = get_chroma_vectors_db(pages)

# get question answer chain

llm = initialize_llm()
qa_chain = get_question_answer_chain(vector_db, llm)

# get user input
prompt = input("Any questions about the pdf?")
llm_response = qa_chain(prompt)
response = process_llm_response(llm_response)
print(response)





