import streamlit as st
from utils import get_pdf_pages, get_chroma_vectors_db, \
                get_question_answer_chain, process_llm_response, initialize_llm


def main():
    st.set_page_config(page_title="PDF Chatbot")
    st.title("PDF chatbot using llama2 LLM")

    # PDF file upload
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file:

        pages = get_pdf_pages(pdf_file)
        b1 = st.button('Get embeddings and build chroma vectors db')

        if b1:
            if "vector_db" not in st.session_state:
                with st.status("Extracting embeddings and building chroma vectors db using llama2...", expanded=True) as status:
                    st.write("Extracting content from the pdf...")
                    pages = get_pdf_pages(pdf_file)
                    st.write("Creating chroma db...")
                    vector_db = get_chroma_vectors_db(pages)
                    st.write("Initializing llama2 LLM...")
                    llm = initialize_llm()
                    st.write("Initializing langchain QA chain...")
                    st.session_state['qa_chain'] = get_question_answer_chain(vector_db, llm)
                    status.update(label="Complete!", state="complete", expanded=False)

        if "qa_chain" in st.session_state:

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("Any questions about the pdf?"):
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

            llm_response = st.session_state['qa_chain'](prompt)
            response = process_llm_response(llm_response)
                
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
