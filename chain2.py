from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

SUMMARY_TEMPLATE = """
You are a medical AI assistant that generates case summaries.
Generate a case summary for {question}:
case summary::"""

SUMMARY_PROMPT = PromptTemplate.from_template(SUMMARY_TEMPLATE)

QUESTION_TEMPLATE = """
Given the case summary below, answer the query.
Case Summary: {case_summary}
Query: {question}
Answer:"""

QUESTION_PROMPT = PromptTemplate.from_template(QUESTION_TEMPLATE)

summary = RunnablePassthrough() | SUMMARY_PROMPT | llm | StrOutputParser()

inputs = RunnableParallel(
    case_summary = summary,
    question = RunnablePassthrough()
)

chain = inputs | QUESTION_PROMPT | llm | StrOutputParser()

response = chain.invoke({"question": "Sudden Paralysis in a Boy With Headaches"})
print(response)