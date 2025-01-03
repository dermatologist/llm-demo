import os
from langchain.agents import AgentExecutor, AgentType, create_react_agent
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import PromptTemplate
from langchain import hub
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")


# setup the tools
@tool
def bmi(input: str) -> int:
    """Calculates BMI of a person for given height and weight."""

    ## input is string: "height: 180, weight: 70", parse it to get height and weight
    height = int(input.split(",")[0].split(":")[1].strip().replace("cm", ""))
    weight = int(input.split(",")[1].split(":")[1].strip().replace("kg", ""))
    bmi = weight / ((height/100) ** 2)
    return " Answer: The BMI of a person with height {} cm and weight {} kg is {:.2f}".format(height, weight, bmi)


@tool
def multiply(input: str) -> int:
    """Multiply two numbers."""
    print(input)
    return int(input.split(",")[0]) * int(input.split(",")[1])



tools = [bmi, multiply]
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=1,
    verbose=True
)

# Run the agent
question = "Calculate the BMI of a person with height 180 cm and weight 70 kg."
response = agent_executor.invoke({"input": question})


print(response)