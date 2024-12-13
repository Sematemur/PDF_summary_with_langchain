from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

load_dotenv() ##environmenti yüklüyoruz.

st.title("PDF SUMMARIZATION") #streamlit ile title oluşturduk

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF") ##pdf'i yüklemek için bir bölüm oluşturuyor
summarize_button = st.button("Generate Summary") #özet oluşturmak için butona basılması lazım

if summarize_button and uploaded_file:
    with st.spinner("Fetching PDF content..."):
        # yüklenen pdf'i bir path haline getiriyoruz çünkü pypdfloader path ile çalışan bir fonksiyon.
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name  

        loader = PyPDFLoader(temp_file_path) #oluşturulan pdf path'ini burada kullandık
        documents = loader.load_and_split() 
        st.write("**PDF content fetched successfully!**")

       #chunklara ayırma işlemini burada yaptık
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        st.write(f"Document divided into {len(chunks)} chunks.")

        api_key = os.getenv("GROQ_API_KEY") #api key'imizi oluşturuyoruz.
        model = ChatGroq(model="llama3-8b-8192", temperature=0.7,api_key=api_key)

        summarize_chain = load_summarize_chain(llm=model, chain_type="map_reduce", verbose=True)##load_summarize_chain: bu fonk kendisi özet çıkarıyor.
        
        with st.spinner("Generating summary..."):
            summary = summarize_chain.run(chunks)
        
        st.write("**Summary of the PDF:**")
        st.write(summary)

# Styling for the app
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #f1f3f4;
        }
        .stButton>button {
            background-color: #6200EE;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #3700B3;
        }
    </style>
""", unsafe_allow_html=True)
