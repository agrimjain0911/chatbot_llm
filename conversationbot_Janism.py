import warnings
warnings.filterwarnings("ignore")
import requests
import yfinance as yf
import re

from typing import TypedDict, List, Optional, Dict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel, ValidationError
from langchain.document_loaders import PyPDFLoader
from datetime import datetime
from langchain_core.messages import SystemMessage

import pandas as pd
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import httpx
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

current_date = datetime.now().strftime("%Y-%m-%d")
from langchain_openai import ChatOpenAI

FORBIDDEN_PATTERNS = [
    r"ignore.*system",
    r"reveal.*prompt",
    r"training.*cutoff",
    r"model.*freeze",
]

def CreateDocuments(records):
    documents = []

    for r in records:
        print(r)
        text = " | ".join(
            f"{k}: {v}" for k, v in r.items()
        )

        documents.append(
            Document(
                page_content=text
            )
        )
    print(f"✅ Loaded {len(documents)} documents")
    return documents

def input_guardrail(user_input: str):
    text = user_input.lower()
    for p in FORBIDDEN_PATTERNS:
        if re.search(p, text):
            return {
                "blocked": True,
                "response": "This request is not allowed."
            }
    return {"blocked": False}


def metadata_guardrail(user_input: str):
    keywords = ["training cutoff", "knowledge cutoff", "model freeze"]
    if any(k in user_input.lower() for k in keywords):
        return {
            "blocked": True,
            "response": "Training cutoff not publicly disclosed."
        }
    return {"blocked": False}


class AgentOutput(BaseModel):
    content: str = ""
    action: str = "none"  # default action to avoid missing field
    tool: Optional[Dict] = None
    tool_input: Optional[Dict] = None

def checkthreshold(vectorstore,query):
    docs_scores = vectorstore.similarity_search_with_score(query, k=5)
    THRESHOLD = 0.7
    relevant_docs = [
        doc for doc, score in docs_scores
        if score < THRESHOLD
    ]

    return relevant_docs

def output_guardrail(llm_output: dict) -> AgentOutput:
    """
        Enforces that the agent only returns allowed structured output.
        Any free-form or missing fields get converted to an error.
        """
    # Ensure required keys exist
    safe_output = {
        "content": llm_output.get("content", ""),
        "action": llm_output.get("action", "none"),  # fix missing field
        "tool": llm_output.get("tool"),
        "tool_input": llm_output.get("tool_input")
    }
    return AgentOutput(**safe_output)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def CreatVector(pdf_file):
    loader = PyPDFLoader(pdf_file)  # <-- your PDF file
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(pdf_file.split('.')[-2])


@tool("pdf_bhatamber")
def pdf_bhatamber(query: str) -> str:
    """
    Use this tool to answer questions about the jain bhatamber and janism teaching of shlokas jain literature.
     """
    print("start: Inside pdf_search with query {0}".format(query))
    vectorstore = FAISS.load_local(folder_path="Bhaktamar_merged_merged", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=10,
        fetch_k=10
    )

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    # docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

@tool("pdf_upnishand")
def pdf_upnishand(query: str) -> str:
    """
    Use this tool to answer questions about the upnishands and lessages from Sanatan Dharma religion.
     """
    print("start: Inside pdf_search with query {0}".format(query))
    vectorstore = FAISS.load_local(folder_path="108upanishads", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    # docs = retriever.get_relevant_documents(query)
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=10,
        fetch_k=20
    )

    return "\n\n".join(d.page_content for d in docs)

@tool("pdf_Geeta")
def pdf_Geeta(query: str) -> str:
    """
      Use this tool to answer questions Sanatan Dharma Geeta and lessons.
    """
    print("start: Inside pdf_search with query {0}".format(query))
    """Get holiday details and company gift policy and process using India_Holidays_and_Gift_Policy.pdf"""
    vectorstore = FAISS.load_local(folder_path="Bhagavad-gita-As-It-Is", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    # docs = retriever.get_relevant_documents(query)
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=10,
        fetch_k=20
    )

    return "\n\n".join(d.page_content for d in docs)

@tool("pdf_aagams")
def pdf_aagams(query: str) -> str:
    """
      Use this tool to answer questions jain aagams and jainism teaching.
    """
    print("start: Inside pdf_search with query {0}".format(query))
    """Get holiday details and company gift policy and process using India_Holidays_and_Gift_Policy.pdf"""
    vectorstore = FAISS.load_local(folder_path="agama_023168_hr6", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    # docs = retriever.get_relevant_documents(query)
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=10,
        fetch_k=20
    )
    threshold = 0.6

    return "\n\n".join(d.page_content for d in docs)

class AgentState(TypedDict):
    messages: List[BaseMessage]


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return "__end__"


def build_graph(llm, tools):
    graph = StateGraph(AgentState)
    react_agent = create_react_agent(llm, tools)
    tool_node = ToolNode(tools)
    graph.add_node("agent", react_agent)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


from langchain_core.messages import HumanMessage


class AgentResponse(BaseModel):
    content: str
    tool: Optional[str] = None
    tool_input: Optional[Dict] = None


def start_chat(query: str, session_id: str, api_key: str,platform) -> str:
    for guard in (input_guardrail, metadata_guardrail):
        result = guard(query)
        if result["blocked"]:
            return result["response"]

    if platform.lower() == "groq":
        llm = ChatGroq(model='moonshotai/kimi-k2-instruct-0905',
                       groq_api_key=os.environ["GROQ_API_KEY"],
                       http_client=httpx.Client(verify=False), )
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7,api_key=os.environ["OPENAI_API_KEY"])

    tools = [
        pdf_bhatamber,
        pdf_upnishand,
        pdf_Geeta,
        pdf_aagams
    ]
    chat_graph = build_graph(llm, tools)

    system_prompt = '''
        Answer ONLY using the provided context.
        Do NOT use prior knowledge and hallucinated answers. 
        give answer atleast 50 words.
        try to give answer in bulleted form.
        Answer format:
        - Answer:
    '''

    if not hasattr(start_chat, "_sessions"):
        start_chat._sessions = {}

    if session_id not in start_chat._sessions:
        start_chat._sessions[session_id] = []

    messages = start_chat._sessions[session_id]

    if system_prompt:
        messages = messages + [SystemMessage(content=system_prompt)]

    messages = messages + [HumanMessage(content=query)]

    try:
        result = chat_graph.invoke({"messages": messages})
    except Exception as e:
        raise
            # if "429" in str(e) or "rate limit" in str(e).lower() or "request too large" in str(e).lower():
            #     return "Rate Limit exceeded or context is too large. please try after sometime with limited context"
            # else:
            #     raise

    # update session memory
    start_chat._sessions[session_id] = result["messages"]

    tools_used = []
    for msg in result["messages"]:
        # Only look at messages that used a tool
        if hasattr(msg, "tool") and msg.tool:
            tools_used.append({
                "tool": msg.tool
            })

    last_content = result["messages"][-1].content if result["messages"] else ""

    parsed = output_guardrail({"content": last_content, "tools_used": tools_used})

    return parsed.content

























