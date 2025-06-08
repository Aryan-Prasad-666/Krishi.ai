import os
import base64
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool
from pydantic import Field
import json

# === Load API Keys ===
load_dotenv()
gemini_api = os.getenv('GEMINI_API_KEY')
serper_key = os.getenv('SERPER_API_KEY')

image_path = r"C:\Users\Aryan Prasad\Downloads\potato-early-blight-leaves.jpg"


# === Base64 Image Encoder ===
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


class CropDiseaseAPI(BaseTool):
    name: str = Field(default="CropDiseaseDetectionAPI", description="Tool to detect crop diseases from image using external ML API.")
    description: str = Field(default="Identifies crop disease by sending base64 image to susya.onrender.com API.")

    def _run(self, image_path: str) -> str:
        try:
            imgdata = encode_image_to_base64(image_path)
            response = requests.post("https://susya.onrender.com", json={"image": imgdata})
            response.raise_for_status()

            # Parse JSON response string
            data = json.loads(response.text)

            disease = data.get("disease", "Unknown disease")
            plant = data.get("plant", "Unknown plant")

            return f"Disease: {disease}\nPlant: {plant}"

        except Exception as e:
            return f"Error calling Crop Disease API: {e}"


# === Instantiate the Tool ===
crop_disease_tool = CropDiseaseAPI()

# === LLM Setup ===
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=gemini_api
)

# === Agents ===

# Agent 1: Uses the API tool to detect disease
crop_disease_identifier = Agent(
    role="Gemini Pathologist",
    goal="Identify diseases in crops using image analysis tools.",
    backstory="An AI crop doctor that uses ML APIs to detect diseases from crop images.",
    tools=[crop_disease_tool],
    verbose=True,
    llm=llm
)

# Agent 2: Suggests remedies
remedy_advisor = Agent(
    role="Agro Remedy Consultant",
    goal="Suggest effective solutions for crop diseases.",
    backstory="An expert agronomist helping farmers treat plant infections.",
    verbose=True,
    llm=llm
)

# Agent 3: Web search expert
serper_tool = SerperDevTool(
    api_key=serper_key,
    n_results=3
)

resource_link_finder = Agent(
    role="Agro Web Researcher",
    goal="Find helpful guides and links about crop disease treatments.",
    backstory="An AI assistant with access to the web for agricultural research.",
    tools=[serper_tool],
    verbose=True,
    llm=llm
)

# === Image Path ===

# === Tasks ===

disease_identification_task = Task(
    description=f"Use the CropDiseaseDetectionAPI tool to identify the disease in the crop image at path: {image_path}. Return the name of the disease and a brief description.",
    expected_output="Disease name and symptoms.",
    agent=crop_disease_identifier
)

remedy_task = Task(
    description="Suggest remedies, both natural and chemical, to cure the identified crop disease.",
    expected_output="Cure steps, materials needed, and prevention advice.",
    agent=remedy_advisor
)

resource_links_task = Task(
    description="Search the internet for tutorials, guides, or PDFs on how to treat the identified crop disease.",
    expected_output="3-5 useful links with brief summaries.",
    agent=resource_link_finder
)

# === Crew Setup ===
crew = Crew(
    agents=[crop_disease_identifier, remedy_advisor, resource_link_finder],
    tasks=[disease_identification_task, remedy_task, resource_links_task],
    verbose=True
)

# === Run ===
result = crew.kickoff()
print("\n=== Final Output ===")
print(result)
