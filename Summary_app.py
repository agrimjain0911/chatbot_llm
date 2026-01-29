# app.py
import os
import streamlit as st
import pandas as pd
import httpx
from math import ceil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

st.set_page_config(page_title="HR Interview Summary", layout="wide")
st.title("📊 HR Interview Data Summarizer")

VECTORSTORE_PATH = "faiss_index"

# --- Sidebar settings ---
st.sidebar.header("Settings")
batch_size_input = st.sidebar.number_input("Batch Size", min_value=1, max_value=100, value=5)
chunk_size_input = st.sidebar.number_input("Chunk Size", min_value=1000, max_value=100000, value=40000)
chunk_overlap_input = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=5000, value=300)

st.sidebar.header("LLM Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")  # User provides key

st.sidebar.header("Prompts")
batch_prompt = st.sidebar.text_area(
    "Batch Summarization Prompt",
    value='''You are an HR analytics assistant. Summarize this batch concisely, covering:
- Total employees
- Average interview count
- Top 3 and bottom 3 employees by interview count
- Trends related to experience
Batch Data hidden for confidentiality.
''',
    height=200
)
batch_prompt = batch_prompt +'{BATCH_DATA}'

final_prompt_template = st.sidebar.text_area(
    "Final Summary Prompt",
    value='''You are an HR analytics assistant. You have summarized multiple batches of employee interview data (data hidden for confidentiality).
Combine the batch summaries into a single concise report, covering:
- Total employees
- Overall average interview count
- Employees with highest and lowest interview counts
- Trends or patterns across all batches
- Key insights and recommendations for HR''',
    height=200
)

# --- Functions ---
def CreateDocuments(records, batch_size=30):
    documents = []
    text = ""
    i = 0
    for r in records:
        text += " | ".join(f"{k}: {v}" for k, v in r.items()) + "\n"
        i += 1
        if i >= batch_size:
            documents.append(Document(page_content=text))
            text = ""
            i = 0
    if text:
        documents.append(Document(page_content=text))
    st.write(f"✅ Created {len(documents)} documents from {len(records)} records")
    return documents

def load_or_create_vectorstore(chunks, embeddings):
    if os.path.exists(VECTORSTORE_PATH):
        st.info("Loading existing vectorstore from disk...")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        st.info("Creating new vectorstore and saving to disk...")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def summarize_batches(llm, chunks, prompt_template, batch_size=50):
    batch_summaries = []
    total_batches = ceil(len(chunks) / batch_size)
    st.info(f"Summarizing {len(chunks)} chunks in {total_batches} batches...")
    for i in range(total_batches):
        batch = chunks[i * batch_size:(i + 1) * batch_size]
        batch_text = "\n\n".join(batch)
        prompt = prompt_template.replace("{BATCH_DATA}", batch_text)
        response = llm.invoke(prompt)
        batch_summaries.append(response.content)
        st.write(f"✅ Completed batch {i+1}/{total_batches}")
    return batch_summaries

# --- Main ---
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    if not groq_api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
    else:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.info("Reading uploaded Excel file...")
        try:
            df_interviews = pd.read_excel(uploaded_file, sheet_name="Interview_Count")
            df_training = pd.read_excel(uploaded_file, sheet_name="Training_Data")
        except Exception as e:
            st.error(f"Error reading Excel sheets: {e}")
            st.stop()

        merged_df = pd.merge(df_interviews, df_training, on="Emp_ID", how="inner")

        st.subheader("Select columns for summary")
        selected_columns = st.multiselect(
            "Choose columns to include in summarization:",
            options=merged_df.columns.tolist(),
        )

        if selected_columns:
            filtered_df = merged_df[selected_columns]
            st.write("Preview of selected data:")
            st.dataframe(filtered_df.head())

            if st.button("Run Summary"):
                st.write(f"Total records: {len(filtered_df)}")
                docs = CreateDocuments(filtered_df.to_dict(orient="records"), batch_size=30)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size_input, chunk_overlap=chunk_overlap_input
                )
                chunks = []
                for doc in docs:
                    chunks.extend(splitter.split_text(doc.page_content))
                st.write(f"Total chunks after splitting: {len(chunks)}")

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = load_or_create_vectorstore(chunks, embeddings)

                llm = ChatGroq(
                    model='moonshotai/kimi-k2-instruct-0905',
                    groq_api_key=groq_api_key,
                    http_client=httpx.Client(verify=False),
                )

                # Batch summarization
                batch_summaries = summarize_batches(llm, chunks, batch_prompt, batch_size=batch_size_input)

                # Final summary
                final_summary = llm.invoke(final_prompt_template)
                st.subheader("📄 Final Summary")
                st.write(final_summary.content)
