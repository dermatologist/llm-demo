from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

TEMPLATE = """
Context information is below.
 --------------------
 Katie presents today's session feeling “depressed.” She states that her mood has been “getting worse.”
 She reports that she has been struggling to get out of the house to enjoy social events that she is actually interested in,
 then becomes more frustrated with herself. She describes an increase in sleep issues,
 both difficulty falling asleep and staying asleep. Katie says that she “just wants to be better.”
 Katie continues to endorse low energy, loss of interest in activities, difficulty sleeping, and feelings of guilt,
 although she is not able to express the source of her guilt.
 --------------------
 Given the context information and not prior knowledge, answer the query.
 Query: {question}

 Answer:
"""

PROMPT = PromptTemplate.from_template(TEMPLATE)

_input = {
    "question": "What are Katie's complaints?"
}

chain = PROMPT | llm | StrOutputParser()

response = chain.invoke(_input)
print(response)