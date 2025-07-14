from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile, os, uuid, shutil
from dotenv import load_dotenv
import time

# PDF and text processing
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pdf2image
import pytesseract
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# API keys
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ONLINE_OCR_API = os.getenv("ONLINE_OCR_API")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment.")
genai.configure(api_key=GEMINI_API_KEY)

# Tesseract setup
TESSERACT_PATHS = [
    '/opt/homebrew/bin/tesseract',
    '/usr/local/bin/tesseract',
    '/usr/bin/tesseract'
]
for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

# In-memory session storage
sessions = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_with_pytesseract(pdf_path):
    images = pdf2image.convert_from_path(pdf_path)
    extracted_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n"
    return extracted_text.strip()

def process_with_online_ocr(pdf_content, api_key):
    api_url = "https://api.ocr.space/parse/image"
    payload = {
        'apikey': api_key,
        'language': 'eng',
        'isOverlayRequired': 'false',
        'OCREngine': '2'
    }
    files = {
        'file': ('document.pdf', pdf_content, 'application/pdf')
    }
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    response = session.post(api_url, files=files, data=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    if result.get('IsErroredOnProcessing'):
        raise HTTPException(status_code=400, detail=f"OCR API Error: {result.get('ErrorMessage')}")
    extracted_text = ""
    for parsed_result in result.get('ParsedResults', []):
        extracted_text += parsed_result.get('ParsedText', '') + "\n"
    return extracted_text.strip()

def get_pdf_text(pdf_files: List[UploadFile]):
    text = ""
    for pdf in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(pdf.file, tmp_file)
            tmp_file_path = tmp_file.name
        try:
            pdf_reader = PdfReader(tmp_file_path)
            page_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            if not page_text.strip():
                ocr_text = process_with_pytesseract(tmp_file_path)
                if not ocr_text and ONLINE_OCR_API:
                    with open(tmp_file_path, 'rb') as f:
                        pdf_bytes = f.read()
                    ocr_text = process_with_online_ocr(pdf_bytes, ONLINE_OCR_API)
                text += ocr_text or ""
            else:
                text += page_text
        finally:
            os.unlink(tmp_file_path)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    return InMemoryVectorStore.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversational_chain():
    prompt_template = """
    You are a helpful AI assistant that can ONLY answer questions using the information from the provided context.
    Do NOT use any pre-trained knowledge or external information.
    If the answer is not available in the provided context, respond with: "This information is not available in the provided documents."
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GEMINI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    raw_text = get_pdf_text(files)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in uploaded PDFs.")
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "vector_store": vector_store,
        "chat_history": []
    }
    return {"session_id": session_id}

@app.post("/ask")
async def ask_question(session_id: str = Form(...), question: str = Form(...)):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    vector_store = session["vector_store"]
    docs = vector_store.similarity_search(question)
    if not docs:
        return {"answer": "This information is not available in the provided documents."}
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    answer = response["output_text"]
    session["chat_history"].append({"role": "user", "content": question})
    session["chat_history"].append({"role": "assistant", "content": answer})
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"Hello": "World"} 