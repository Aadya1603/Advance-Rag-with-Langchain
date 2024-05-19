import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_olla import OLLAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Create API wrappers
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)

# Create query runs
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Load documents
loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)

# Create vector database
vectordb=FAISS.from_documents(documents,OLLAEmbeddings())

# Create retriever
retriever=vectordb.as_retriever()

# Create retriever tool
retriever_tool=create_retriever_tool(retriever,"langsmith_search",
                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

# Create tools list
tools=[wiki,arxiv,retriever_tool]

# Create LLM
llm = openai.ChatCompletion(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125", temperature=0)

# Create prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create agent
agent=create_openai_tools_agent(llm,tools,prompt)

# Create agent executor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

# Streamlit app
st.title("LangSmith Agent")
st.write("Ask me anything about LangSmith!")

input_text = st.text_input("Enter your question:")

if st.button("Ask"):
    output = agent_executor.invoke({"input": input_text})
    st.write("Answer:", output)