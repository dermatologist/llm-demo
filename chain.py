from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

TEMPLATE = """
You are a medical AI assistant that generates case summaries.
Generate a case summary for {question}:
case summary::"""

PROMPT = PromptTemplate.from_template(TEMPLATE)

chain = PROMPT | llm | StrOutputParser()
response = chain.invoke({"question": "Sudden Paralysis in a Boy With Headaches"})
print(response)