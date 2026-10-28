import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

from langchain.agents import create_react_agent, AgentExecutor

# ======================
#   load env variables
# ======================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearchResults(max_results=2)

result = search_tool.invoke("Give me the latest news on AI")
result

# ======================
# LLM
# =====================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY,
)

response =llm.invoke("What position does CR7 play?")
response

# PROMPT
# ======================

prompt = hub.pull("hwchase17/react")

tools = [search_tool]

# ======================
# create agent
# ======================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
 
)

# ======================
# Executor
# ======================    
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  

# ======================
# RUN
# ====================
response = agent_executor.invoke({ 
    "input": ("Find the capital of India")
})    