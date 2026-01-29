import warnings
warnings.filterwarnings("ignore")

import httpx
import requests
import yfinance as yf
import os

from typing import TypedDict, List

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from typing import List
from pydantic import BaseModel, ValidationError
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class AgentOutput(BaseModel):
    action: str
    tool_name: str | None = None
    tool_input: dict | None = None

def output_guardrail(llm_output: dict):
    return AgentOutput(**llm_output)

def CreatVector(pdf_file):
    loader = PyPDFLoader(pdf_file)   # <-- your PDF file
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("pdf_vectorstore")

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
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

def build_graph(llm,tools):
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

def start_chat(query: str, session_id: str, api_key: str) -> str:
    llm = ChatGroq(model='moonshotai/kimi-k2-instruct-0905',
                   groq_api_key=os.environ["GROQ_API_KEY"],
                   http_client=httpx.Client(verify=False))
    tools = [
        pdf_knowledge_base,
        web_search,
        get_stock_info,
        get_dividends
    ]
    chat_graph = build_graph(llm,tools)

    # 🔹 In-memory session store (replaces Redis)
    if not hasattr(start_chat, "_sessions"):
        start_chat._sessions = {}

    if session_id not in start_chat._sessions:
        start_chat._sessions[session_id] = []

    messages = start_chat._sessions[session_id]

    messages = messages + [HumanMessage(content=query)]

    result = chat_graph.invoke({"messages": messages})

    # update session memory
    start_chat._sessions[session_id] = result["messages"]

    return result["messages"][-1].content



