from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from duckduckgo_search import DDGS
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os

# Set up LangChain project and environment
os.environ['LANGCHAIN_PROJECT'] = 'Agent Trace'
load_dotenv()

# ------------------ SEARCH TOOL (using ddgs) ------------------

@tool
def duckduckgo_search(query: str) -> str:
    """
    Performs a DuckDuckGo web search using the new ddgs package.
    Returns the top 3 results with title, link, and snippet.
    """
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
    formatted = "\n\n".join(
        [f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results]
    )
    return formatted or "No results found."

# ------------------ WEATHER TOOL ------------------

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches current weather data for a given city using Weatherstack API.
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    data = response.json()

    if "current" in data:
        temp = data["current"]["temperature"]
        desc = data["current"]["weather_descriptions"][0]
        return f"The current temperature in {city} is {temp}°C with {desc.lower()}."
    else:
        return f"Could not fetch weather data for {city}."

# ------------------ LLM and AGENT SETUP ------------------

llm = ChatOpenAI(model="gpt-4o-mini")  # or any available OpenAI model

# Pull ReAct prompt template from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the agent with our tools
agent = create_react_agent(
    llm=llm,
    tools=[duckduckgo_search, get_weather_data],
    prompt=prompt
)

# Wrap the agent in an executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[duckduckgo_search, get_weather_data],
    verbose=True,
    max_iterations=5
)

# ------------------ TEST THE AGENT ------------------

# Example: web + weather queries
queries = [
    #"What is the release date of Dhadak 2?",
    "What is the current temperature in hyderabad?",
    #"Identify the birthplace city of Kalpana Chawla (search) and give its current temperature."
]

for q in queries:
    print(f"\n🧠 Query: {q}")
    response = agent_executor.invoke({"input": q})
    print("🗣️ Response:", response["output"])
