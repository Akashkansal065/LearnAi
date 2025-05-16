from markitdown import MarkItDown
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import ollama
from ollama import Client

ollamaclient = Client(host="http://localhost:11434/api/generate")
md = MarkItDown(llm_client=ollamaclient,
                llm_model="laama3.2-vision", enable_plugins=True)
# result = md.convert(
# '/Users/akash.kansal/Downloads/OPD - Claim Form_22042025141246.pdf')
result = md.convert(
print(result.text_content)
load_dotenv()  # Load environment variables

# -----------------------------------------------
# Step 1: Define your Pydantic response model
# -----------------------------------------------


class CalendarEvent(BaseModel):
    name: str = Field(description="The name of the event")
    date: str = Field(description="The date of the event")
    participants: list[str] = Field(description="People involved")


parser = PydanticOutputParser(pydantic_object=CalendarEvent)

# -----------------------------------------------
# Step 2: Set up the LangChain Ollama LLM
# -----------------------------------------------

llm = ChatOllama(model="phi4", base_url="http://localhost:11434")

# -----------------------------------------------
# Step 3: Create Prompt Template with Format Instructions
# -----------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "Extract the event information and return it in structured format."},
        {
            "role": "user",
            "content": "{input}\n\n{format_instructions}",
        }])

formatted_prompt = prompt.format_messages(
    input="Alice and Bob are going to a science fair on Friday.",
    format_instructions=parser.get_format_instructions()
)

# -----------------------------------------------
# Step 4: Generate and Parse the Output
# -----------------------------------------------

response = llm.invoke(formatted_prompt)

parsed = parser.parse(response.content)

print(parsed)
print("Event name:", parsed.name)
print("Date:", parsed.date)
print("Participants:", parsed.participants)
