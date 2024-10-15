import os

from fastapi import FastAPI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI()

class Query(BaseModel):
    question: str

class Query_Response(BaseModel):
    question: str
    response: str

@app.get("/")
async def root():
    return {"message": "Hello Render"}

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct Postgres query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""





@app.post("/question")
async def question(q: Query):
    question = q.question
    system_message = SystemMessage(content=SQL_PREFIX)

    db = SQLDatabase.from_uri(os.environ['DB_URL'])
    print(db.dialect)
    print(db.get_usable_table_names())
    llm = ChatOpenAI(model="gpt-4o-mini")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)
    final_state = agent_executor.invoke({"messages": question})
    res = final_state["messages"][-1].content

    query_response = Query_Response(question=question, response=res)

    return query_response
