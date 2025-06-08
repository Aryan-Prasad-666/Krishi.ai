from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.google import Gemini
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

llm = Gemini(
    id='gemini-2.0-flash',
    api_key=gemini_api_key,
    temperature=0.7
)

agent = Agent(
    model=llm,
    description="You are an expert researcher using DuckDuckGo to find the latest information.",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    memory=False
)

agent.print_response("Find 5 Indian government schemes for female entrepreneurs in Karnataka.", stream=True)
