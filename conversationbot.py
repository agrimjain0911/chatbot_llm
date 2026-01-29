from langchain.agents import AgentExecutor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*PydanticDeprecatedSince20.*"
)
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
#from langchain_classic.agents import AgentExecutor
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.tools import tool
import httpx
from langchain_core.prompts import PromptTemplate
import json
import requests
from duckduckgo_search import DDGS
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
import httpx
import yfinance as yf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory




llm = ChatGroq(model='moonshotai/kimi-k2-instruct-0905',groq_api_key='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',http_client=httpx.Client(verify=False))

# pdf_file = "India_Holidays_and_Gift_Policy.pdf"
# loader = PyPDFLoader(pdf_file)   # <-- your PDF file
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
# docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore.save_local("pdf_vectorstore")


def pdf_search(query: str) -> str:
    print("start: Inside pdf_search with query {0}".format(query))
    """Get holiday details and company gift policy and process using India_Holidays_and_Gift_Policy.pdf"""
    vectorstore = FAISS.load_local(folder_path="pdf_vectorstore", embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in docs)


def get_dividends(symbol: str) -> str:
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


def get_stock_info(ticker: str) -> str:
    """Get stock price and company info using yfinance"""
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


def web_search(query: str) -> str:
    """Search the web for latest information"""
    print("start: Inside web_search with query {0}".format(query))
    with DDGS(timeout=20) as ddgs:
        results = list(ddgs.text(query, max_results=3))
    return str(results)

pdf_tool = Tool(
    name="PDF Knowledge Base",
    func=pdf_search,
    description=(
        "Use this tool to answer questions about the company holiday list "
        "and gift policy document."
    )
)

API_tool = Tool(
    name="Dividend details of stock",
    func=get_dividends,
    description=(
        "Get dividend information for a stock symbol"
    )
)

search_tool = Tool(
    name="Web Search",
    func=web_search,
    description="Search the web for current events and news"
)

finance_tool = Tool(
    name="Stock Info",
    func=get_stock_info,
    description="Get stock price and company financial info using ticker symbol")


def create_agent(session_id: str):
    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True
    )
    generic_agent = initialize_agent(
        tools=[pdf_tool, search_tool, finance_tool,API_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )
    return generic_agent


def start_chat(query,session_id: str):
    generic_agent = create_agent(session_id)
    response = generic_agent.invoke(query)
    return response['output']

