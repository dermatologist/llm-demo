from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

TEMPLATE = """
Artificial Intelligence (AI) has surpassed dermatologists in skin cancer detection, but dermatology still lags behind radiology in its broader adoption. Building and using AI applications are becoming increasingly accessible. However, complex use cases may still require specialized expertise for design and deployment. AI has many applications in dermatology ranging from fundamental research, diagnostics, therapeutics, and cosmetic dermatology. The lack of standardization of images and privacy concerns are the foremost challenges stifling AI adoption. Dermatologists have a significant role to play in standardized data collection, curating data for machine learning, clinically validating AI solutions, and ultimately adopting this paradigm shift that is changing the way we practice.
\n\n

Answer the following question from the text above.\n\nQ: {question}?\nA::



"""

PROMPT = PromptTemplate.from_template(TEMPLATE)

_input = {
    "question": "What are the foremost challenges stifling AI adoption in dermatology"
}

chain = PROMPT | llm | StrOutputParser()

response = chain.invoke(_input).split("::")[-1].strip()
print(response)