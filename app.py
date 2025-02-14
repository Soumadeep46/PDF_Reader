from dotenv import load_dotenv
import os
import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import faiss
import numpy as np

# Load API Key
load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_api_token:
    st.error("Hugging Face API Token is missing. Please set HUGGINGFACEHUB_API_TOKEN in the .env file.")

# Function to extract text from a PDF
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Failed to extract text: {str(e)}")
        return None

# Function to chunk and embed text
@st.cache_data
def chunk_and_embed_text(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embedded_chunks = embeddings.embed_documents(chunks)

        return chunks, embedded_chunks
    except Exception as e:
        st.error(f"Failed to chunk or embed text: {str(e)}")
        return None, None

# Function to set up FAISS index
@st.cache_resource
def setup_faiss_index(embedded_chunks):
    try:
        vectors = np.array(embedded_chunks).astype('float32')
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return index
    except Exception as e:
        st.error(f"Failed to set up FAISS index: {str(e)}")
        return None

# Function to query FAISS index
def query_faiss_index(index, query_embedding, k=1):
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    return indices[0]

# Summarization using Hugging Face Model
def summarize_content(text):
    try:
        llm = HuggingFaceHub(
            repo_id="facebook/bart-large-cnn",
            model_kwargs={"temperature": 0.2, "max_length": 250},
            huggingfacehub_api_token=huggingface_api_token
        )
        summary = llm(f"Summarize: {text}")
        return summary
    except Exception as e:
        st.error(f"Summarization failed: {str(e)}")
        return None

# Setup LLM for querying
# Fix the setup_llm function
def setup_llm():
    try:
        llm = HuggingFaceHub(
            repo_id='mistralai/Mistral-7B-Instruct-v0.1',
            model_kwargs={"temperature": 0.5, "max_length": 1000},
            huggingfacehub_api_token=huggingface_api_token
        )
        return llm
    except Exception as e:
        st.error(f"Failed to setup LLM: {str(e)}")
        return None

# Chat response based on context
def chat_response(llm, user_input, context):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Context: {context}
        Question: {question}
        Answer:"""
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"context": context, "question": user_input})
    
    # Extract just the answer part, removing any prefixes
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    
    return response

# Streamlit app

def main():
    st.title("📚 AI Study Assistant (Hugging Face)")
    uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            with st.spinner("Chunking and Embedding Text..."):
                chunks, embedded_chunks = chunk_and_embed_text(pdf_text)

            if chunks and embedded_chunks:
                with st.spinner("Setting up FAISS Index..."):
                    faiss_index = setup_faiss_index(embedded_chunks)

                if faiss_index is not None:
                    st.success("✅ PDF Processed Successfully!")

                    # Setup LLM
                    llm = setup_llm()
                    if llm is None:
                        st.error("Failed to initialize the language model.")
                        return

                    user_input = st.text_input("💬 Ask a question about the PDF:")
                    if user_input:
                        try:
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                            query_embedding = embeddings.embed_query(user_input)
                            similar_chunk_indices = query_faiss_index(faiss_index, query_embedding)
                            context = chunks[similar_chunk_indices[0]]
                            response = chat_response(llm, user_input, context)

                            st.markdown("### 🧑‍💼 **User Query:**")
                            st.info(user_input)

                            st.markdown("### 📄 **Context Extracted:**")
                            st.write(f"> {context}")

                            st.markdown("### 🤖 **AI Response:**")
                            st.markdown(
                                f"""<div style='
                                    background-color: #003300; 
                                    padding: 20px; 
                                    border-radius: 5px; 
                                    color: #b9fbc0;
                                    white-space: normal;
                                    word-wrap: break-word;
                                    margin: 10px 0;'
                                >{response}</div>""", 
                                unsafe_allow_html=True
                            )

                        except Exception as e:
                            st.error(f"Chat processing failed: {str(e)}")

                    if st.button("📄 Summarize Content"):
                        summary = summarize_content(pdf_text[:2000])
                        if summary:
                            st.write("📄 **Summary:**", summary)

if __name__ == "__main__":
    main()
