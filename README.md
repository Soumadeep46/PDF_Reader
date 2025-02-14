# 📚 AI Study Assistant

This project is a PDF-based AI Study Assistant built using Streamlit and Hugging Face models. It allows users to upload a PDF, extract content, ask questions based on the document, and receive answers using contextual embeddings and LLMs. Additionally, users can summarize the content of the uploaded PDF.

## 🚀 Features
- Extract text from uploaded PDF files
- Chunk and embed text using `sentence-transformers/all-MiniLM-L6-v2`
- Create a FAISS index for efficient similarity search
- Answer user questions based on PDF content using `mistralai/Mistral-7B-Instruct-v0.1`
- Summarize content using `facebook/bart-large-cnn`

## 🛠️ Installation

Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone https://github.com/Soumadeep46/PDF_Reader.git
cd PDF_Reader

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate     # On Windows

# Install required packages
pip install streamlit langchain faiss-cpu numpy python-dotenv pymupdf
```

## 🔑 Setup Hugging Face API Key

Create a `.env` file in the root directory and add:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```

You can get the API key from [Hugging Face](https://huggingface.co/settings/tokens).

## ▶️ Running the Application

```bash
streamlit run app.py
```


## 📄 How It Works
1. Upload a PDF file.
2. The app extracts text, splits it into chunks, and embeds them.
3. It builds a FAISS index for similarity search.
4. Ask a question based on the uploaded PDF content.
5. The app retrieves the most relevant chunk and queries the LLM for an answer.
6. You can also summarize the PDF content.

## 🧠 Models Used
| Task                    | Model                                    | Source                                                   |
|-------------------------|-------------------------------------------|-----------------------------------------------------------|
| Embedding               | `sentence-transformers/all-MiniLM-L6-v2` | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Question Answering (LLM)| `mistralai/Mistral-7B-Instruct-v0.1`     | [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)     |
| Summarization           | `facebook/bart-large-cnn`                | [Hugging Face](https://huggingface.co/facebook/bart-large-cnn)                |


## 📝 Notes
- Large PDFs may take time to process.
- Only the first 2000 characters are summarized due to model limitations.

## 🐛 Troubleshooting
- Ensure the Hugging Face API key is set up correctly.
- Install missing packages if any errors occur using:

```bash
pip install huggingface_hub sentence-transformers
```

## 📧 Contact
For any issues or improvements, feel free to create an issue or reach out.

---
Enjoy using your AI-powered study assistant! 🚀

