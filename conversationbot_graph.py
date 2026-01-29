import warnings

warnings.filterwarnings("ignore")

import httpx
import requests
import yfinance as yf
import os
import re

from typing import TypedDict, List, Optional, Dict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel, ValidationError
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from datetime import datetime
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.docstore.document import Document

import pandas as pd
import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import json
from langchain.docstore.document import Document

current_date = datetime.now().strftime("%Y-%m-%d")

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

def CreateVector(excel_file):
    if os.path.exists("PMS25"):
        print("✅ Loading existing embeddings")
        vectorstore = FAISS.load_local(
            "PMS25",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        df_Interviews = pd.read_excel(excel_file, sheet_name="Interview_Count")
        df_Training = pd.read_excel(excel_file, sheet_name="Training_Data")
        merged_df = pd.merge(df_Interviews, df_Training, on="Emp_ID", how="inner")

        print("🆕 Creating embeddings from JSON")
        documents = CreateDocuments(merged_df.to_dict(orient="records"))
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("PMS25")
        print("✅ Embeddings saved")
    return vectorstore
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


ALLOWED_TOOLS = {"pdf_knowledge_base", "web_search", "get_stock_info","PMS_data"}


def tool_guardrail(tool_name: str):
    if tool_name not in ALLOWED_TOOLS:
        raise RuntimeError(f"Tool '{tool_name}' is not allowed")


class AgentOutput(BaseModel):
    content: str = ""
    action: str = "none"  # default action to avoid missing field
    tool: Optional[Dict] = None
    tool_input: Optional[Dict] = None


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
    vectorstore.save_local("pdf_vectorstore")


@tool("pdf_Bhagwat_Geeta")
def pdf_Bhagwat_Geeta(query: str) -> str:
    """
    Use this tool to answer questions about the Geeta and teaching of bhagwat.
     """
    print("start: Inside pdf_search with query {0}".format(query))
    """Get holiday details and company gift policy and process using India_Holidays_and_Gift_Policy.pdf"""
    vectorstore = FAISS.load_local(folder_path="Bhagavad-gita-As-It-Is", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

@tool("PMS_data")
def PMS_data(query: str) -> str:
    """
      Use this tool to answer questions about the PMS data, interview counts, Mandatory training taken, employee name.
    """
    file_path="PMS2025.xlsx"
    vectorstore = CreateVector(file_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)

@tool("pdf_knowledge_base")
def pdf_knowledge_base(query: str) -> str:
    """
      Use this tool to answer questions about the company holiday list and gift policy document.
    """
    print("start: Inside pdf_search with query {0}".format(query))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    """Get holiday details and company gift policy and process using India_Holidays_and_Gift_Policy.pdf"""
    vectorstore = FAISS.load_local(folder_path="pdf_vectorstore", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)


@tool("get_dividends")
def get_dividends(symbol: str) -> str:
    """
    Get dividend information for a stock symbol
    """
    url = "https://api.massive.com/v3/reference/dividends"
    api_key = "q1bvkUCP8TDc__Q_uPLgQL2FupngRprR"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "ticker": symbol
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    response.raise_for_status()
    return response.json()


@tool("get_stock_info")
def get_stock_info(ticker: str) -> str:
    """
    Get stock price and company info using yfinance
    """
    print("start: Inside get stock Info with ticker {0}".format(ticker))
    stock = yf.Ticker(ticker)
    info = stock.info
    print("end : Inside get stock Info with ticker {0}".format(ticker))
    return f"""
    Company: {info.get('longName')}
    Ticker: {ticker}
    Price: {info.get('currentPrice')}
    Market Cap: {info.get('marketCap')}
    Currency: {info.get('currency')}
    """


@tool("web_search")
def web_search(query: str) -> str:
    """
    Search Indian news sources for the latest updates.
    """
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "country": "in",
        "max": 5,
        "apikey": "b84c97e68502419f4465e65fd3f926ca"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    articles = r.json().get("articles", [])
    return "\n\n".join(
        f"{a['title']} — {a['source']['name']}\n{a['url']}"
        for a in articles
    )
    # """
    # Search the web for latest information
    # """
    # print("start: Inside web_search with query {0}".format(query))
    # with DDGS(timeout=20) as ddgs:
    #     results = list(ddgs.text(query, max_results=3))
    # return str(results)


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

    print(platform)

    if platform.lower() == "groq":
        llm = ChatGroq(model='moonshotai/kimi-k2-instruct-0905',
                       groq_api_key=os.environ["GROQ_API_KEY"],
                       http_client=httpx.Client(verify=False), )
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7,api_key=os.environ["OPENAI_API_KEY"])

    tools = [
        pdf_Bhagwat_Geeta,
        pdf_knowledge_base,
        web_search,
        get_stock_info,
        get_dividends,
        PMS_data
    ]
    chat_graph = build_graph(llm, tools)

    system_prompt = '''
        - Today's date is {datetime.now().strftime('%Y-%m-%d')}. Use this as the current date
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

















