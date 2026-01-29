import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import httpx

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def load_llm(groq_api_key: str):
    return ChatGroq(
        model="moonshotai/kimi-k2-instruct-0905",
        groq_api_key=groq_api_key,
        http_client=httpx.Client(verify=False),
        temperature=0
    )


def load_resume_chunks(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)


def get_prompt():
    return PromptTemplate.from_template("""
You are an ATS resume evaluator.

Scoring Rules:
- If ALL required skills are present → 85–95%
- If MOST required skills (≥75%) → 70–84%
- If SOME skills (40–74%) → 40–69%
- If FEW skills (<40%) → below 40%

Experience Rules:
- If experience meets or exceeds JD → do NOT penalize score
- If experience exceeds JD significantly → add +5%

Instructions:
- Compare Job Description skills vs Resume skills
- Count explicit skill matches
- Do NOT lower score due to unrelated resume sections
- Do NOT hallucinate
- Cite resume page numbers
- Reason should show which requirements are matched and which are not and why score penalize

Output Format (STRICT) as a bullet points:
Match Percentage: XX%
Fit Verdict: Yes / No
Reasoning: <min 50 words>

Resume Context:
{context}

Job Description:
{input}

Answer:
""")

def build_rag_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=get_prompt()
    )

    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

def run_resume_match(resume_pdf_path, job_description, groq_api_key):
    llm = load_llm(groq_api_key)
    docs = load_resume_chunks(resume_pdf_path)
    vectorstore = create_vectorstore(docs)

    rag_chain = build_rag_chain(llm, vectorstore)

    return rag_chain.invoke({
        "input": job_description
    })


st.set_page_config(
    page_title="Resume vs JD Matcher (AI)",
    layout="centered"
)

st.title("📄 Resume ↔ Job Description Matcher")
st.write("AI-powered resume screening using RAG + Groq LLM")

st.divider()


groq_api_key = st.text_input(
    "🔑 Enter Groq API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "📄 Upload Resume (PDF only)",
    type=["pdf"]
)

job_description = st.text_area(
    "📝 Paste Job Description",
    height=200,
    placeholder="Enter full job description here..."
)

if st.button("▶️ Match Resume with JD"):
    if not groq_api_key:
        st.error("Please enter Groq API Key.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("Please enter Job Description.")
    else:
        with st.spinner("Analyzing resume..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_pdf_path = tmp.name

            try:
                result = run_resume_match(
                    temp_pdf_path,
                    job_description,
                    groq_api_key
                )

                st.success("✅ Analysis Complete")

                st.subheader("📊 Match Result")
                st.write(result["answer"])
           

            except Exception as e:
                st.error(f"Error: {str(e)}")



