from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

TEMPLATE = """
You are an AI agent that generates product descriptions.

Generate product descriptions for the last product as in the examples below:

Product: Wireless Earbuds

Description: Immerse yourself in crystal-clear audio with our sleek wireless earbuds. Featuring noise-cancellation technology and a comfortable fit, these earbuds are perfect for music lovers on the go.​

Product: Smart Watch

Description: Stay connected and track your fitness with our advanced smart watch. With heart rate monitoring, GPS, and a vibrant touch screen, it's your perfect companion for an active lifestyle.​

Product: {question}

Description::
"""

PROMPT = PromptTemplate.from_template(TEMPLATE)

_input = {
    "question": "Ergonomic Office Chair"
}

chain = PROMPT | llm | StrOutputParser()

response = chain.invoke(_input).split("::")[-1].strip()
print(response)