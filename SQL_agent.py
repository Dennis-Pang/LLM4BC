from dotenv import load_dotenv
import os
from typing import Literal
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from general_functions import create_tool_node_with_fallback, get_next_case_number, get_case_number, State, create_new_state
from sqlalchemy import create_engine
import csv
import psutil
import time
import os
import pandas as pd
from sqlalchemy import create_engine


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

state = create_new_state()
workflow = StateGraph(State)

# 全部使用生成的绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))

results_dir = os.path.join(project_root, 'results')
table_index_dir = os.path.join(project_root, 'dbschema/table_index.txt')
base_path = os.path.join(project_root, 'dbschema/cardano_tables/')


def sql_db_query(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If the query executes successfully, the result will be saved to a new 'case<number>/query_result.csv' file.
    """
    print('"sql_db_query" started (6)')
    try:
        connection_string = "postgresql+psycopg2://map_llm_junyong_zixian:kaj2#snr928jseh34@172.23.50.234:5432/cexplorer"
        engine = create_engine(connection_string)

        # Create the results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Get the current case number
        case_number = get_case_number(results_dir)
        case_dir = f'case{case_number}'
        os.makedirs(os.path.join(results_dir, case_dir), exist_ok=True)

        # Execute the query in chunks and save results in a streaming way
        file_path = os.path.join(results_dir, case_dir, 'query_result.csv')
        with open(file_path, 'w', newline='') as csvfile:
            writer = None
            chunksize = 1000  # Define the chunk size

            for chunk in pd.read_sql_query(query, engine, chunksize=chunksize):
                if writer is None:
                    # Write header only once
                    writer = csv.DictWriter(csvfile, fieldnames=chunk.columns)
                    writer.writeheader()

                # Convert each row of chunk to dictionary and write to CSV
                for _, row in chunk.iterrows():
                    writer.writerow(row.to_dict())

        engine.dispose()
        print(f"Query executed successfully. Results saved to '{file_path}'.")
        return END
    except Exception as e:
        print('error')
        return f"Error: {str(e)}"

@tool
def get_table_list() -> str:
    """
    Use this tool to get the schema of the db
    """
    print('"get_table_list" started (1)')
    with open(table_index_dir, 'r') as file:
        schema_content = file.read()
    return schema_content

@tool
def get_tables_schema(tables_name: list) -> str:
    """
    Use this tool to get the detailed schema of each table
    """
    # Define the path where table metadata is stored
    print('"get_tables_schema" started (3)')
    # Initialize a string to hold the combined content
    combined_content = ""

    # Loop through each table name provided in the list
    for table_name in tables_name:
        # Define the full path to the file corresponding to the table
        file_path = os.path.join(base_path, f"{table_name}.txt")

        # Check if the file exists
        if os.path.exists(file_path):
            # Open and read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Add the content to the combined string
                combined_content += f"\n--- {table_name} ---\n"
                combined_content += content
    case_number = get_next_case_number(results_dir)
    output_directory = os.path.join(results_dir, f'case{case_number}')
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, 'tables_schema.txt')
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(combined_content)
    # Return the combined content
    return combined_content

workflow.add_node("get_tables_schema", create_tool_node_with_fallback([get_tables_schema]))

def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="justfortest",
                tool_calls=[
                    {
                        "name": "get_table_list",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

workflow.add_node("first_tool_call", first_tool_call)

class Information_not_included(BaseModel):
    """Tell the user there is no information needed and make some alternative suggestions based on the schema."""
    Final_answer: str = Field(..., description="The response to the user")

information_check_system = """ 
As a supervisor, your role is to carefully review the database schema to determine whether it contains the necessary information to address the user's query. Follow these guidelines to ensure you provide accurate and useful responses:
You are provided with a list of table names and descriptions in the following format:
Example

tx
A table for transactions within a block on the chain.
tx is the table name.
'A table for transactions within a block on the chain.' is the description.
The get_tables_schema function allows you to retrieve the schema details for a list of tables. Use this function to get the schema for the relevant tables and then generate the SQL queries accordingly.
you need to select relevent tables according to users input and combine them in a list and run the get_tables_schema only once
1. Assess the Database for Relevant Information:
If the database contains relevant information:
Identify the relevant tables and fields that can help address the user's query.
Provide clear guidance on how to construct an SQL query to retrieve the necessary information, specifying the appropriate tables and fields.
If there are any gaps or missing data, clearly explain which parts of the query can be answered and which parts cannot be fulfilled by the database.
If the database does not contain the required information:
Use the Information_not_included tool to inform the user that the database lacks the necessary data.
Offer alternative suggestions based on the available data, such as related queries that might still be useful. Be sure to explain any limitations.
2. Provide Clear Communication:
Ensure all instructions, explanations, and suggestions are clear, concise, and easy to understand.
Offer guidance on how the user can refine their query if the database only partially meets their needs.


If the database lacks the necessary information, use the tool Information_not_included to inform the user. However, if the database contains (fully or partially) the required information, generate the SQL queries directly. Provide guidance solely based on SQL without using any other programming languages.
"""

information_check_prompt = ChatPromptTemplate.from_messages(
    [("system", information_check_system), ("placeholder", "{messages}")]
)
information_check = information_check_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [Information_not_included,get_tables_schema]
)

def information_check_node(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to check if there is enough information. and get relevent tables details.
    """
    print('"information_check" started (2)')

    return {"messages": [information_check.invoke(state)]}

# Removed the SubmitFinalQuery class and references to it

query_check_system = """You are a SQL expert with a strong attention to detail.
Double-check the SQL query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, proceed to execute the query.

You will call the 'sql_db_query' tool to execute the query after running this check.

If the query executes successfully, inform the user that the results have been saved to 'query_result.csv'.

If there is an error during execution, return to the query generation step to regenerate the query and rerun the whole process.
"""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)

query_check = query_check_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [sql_db_query]
)

def check_and_execute_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Checks the SQL query and executes it if correct.
    """
    print('"check_and_execute_query" started (5)')
    return {"messages": [query_check.invoke(state)]}

workflow.add_node("get_table_list", create_tool_node_with_fallback([get_table_list]))
workflow.add_node("information_check", information_check_node)
workflow.add_node("check_and_execute_query", check_and_execute_query)

# Define the query generation node
def query_gen_node(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to generate the SQL query.
    """
    print('"query_gen" started (4)')

    return {"messages": [query_gen.invoke(state)]}

# Update the query generation prompt
query_gen_system = """
You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct and optimized SQL query to run.

When generating the query:

- Output the SQL query that answers the input question without a tool call.
- You can order the results by a relevant column to return the most interesting examples in the database.
- Never query for all the columns from a specific table; only ask for the relevant columns given the question.
- If a WHERE clause is involved, prioritize using indexed columns to enhance query performance.
- When ordering the results, prefer using indexed columns to avoid full table scans.
- Use UNION ALL instead of UNION when duplicates are not a concern, as UNION incurs additional overhead due to the de-duplication step.
- For large result sets, use LIMIT or pagination (OFFSET and LIMIT) to reduce the amount of data retrieved in one query.
- Avoid using subqueries if JOIN can achieve the same result with better performance.
- Always combine queries into one, executing the query in a single call.
- If you get an empty result set, rewrite the query logically to retrieve valid results.
- NEVER make DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.
- If you encounter errors while executing a query, rewrite the query and try again.
- NEVER make up data or results; if there is insufficient information to answer the query, just say you don't have enough information.

DO NOT call any tool besides 'sql_db_query'.

"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o", temperature=0)

workflow.add_node("query_gen", query_gen_node)
workflow.add_node("sql_db_query", create_tool_node_with_fallback([sql_db_query]))

# Define conditional functions
def should_continue_info_check(state: State) -> Literal[END, "query_gen","get_tables_schema"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, and the tool called is 'Information_not_included', then we END
    if getattr(last_message, "tool_calls", None):
        for tc in last_message.tool_calls:
            if tc["name"] == "Information_not_included":
                return END
            elif tc["name"] == "get_tables_schema":
                return "get_tables_schema"
    else:
        return "query_gen"

def should_continue_check_and_execute_query(state: State) -> Literal["query_gen","sql_db_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        for tc in last_message.tool_calls:
            if tc["name"] == "sql_db_query":
                return "sql_db_query"
            elif tc["name"] == "query_gen":
                return "query_gen"
    
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "get_table_list")
workflow.add_edge("get_table_list", "information_check")
workflow.add_edge("get_tables_schema", "information_check")
workflow.add_edge("sql_db_query", END)

workflow.add_conditional_edges(
    "information_check",
    should_continue_info_check,
)


workflow.add_edge("query_gen", "check_and_execute_query")
workflow.add_conditional_edges(
    "check_and_execute_query",
    should_continue_check_and_execute_query,
)

app = workflow.compile()

def get_sql_agent_answer(input: str):
    """Use this for answer evaluation"""
    msg = {"messages": [("user", input)]}
    messages = app.invoke(msg)
    # Assuming the last message contains the result
    response = messages["messages"][-1].content
    return response

'''

# get image
img_data = app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)

# save image
with open("Project_Pythonfile/Agent_pipeline_charts/sql_agent.png", "wb") as file:
    file.write(img_data)
    
'''

def run_sql_agent(input_data):
    final_state = app.invoke(
        {"messages": [("user", input_data)]}
    )
    return final_state

# run_sql_agent("give me the list of latest 10 transactions, their timestamp, their list of inputs and outputs")

# u_prompt="""
# Return a CSV with 10 lines. Each line represents a transaction. 
# Each line includes the 
# 1) "transaction id", 
# 2) "timestamp" of the transaction (should be extracted from the "block" table), 
# 3) list of inputs (for each input, we need the "address" and "value"), and 
# 4) list of outputs (for each output, we need the "address" and "value").

# """
# run_sql_agent(u_prompt)


if __name__ == "__main__":
    u_prompt_median_regression = """
    Return a CSV, where each line represents a validating pool in epoch 280. Each line includes the 
    1) pool id
    2) total stakes delegated to that pool 
    3) total rewards that pool received
    """

    u_prompt_gini_stake = """
    Return a CSV that contains the amount of stake delegated to each active validating pool 
    (pools that have received non-zero stake in an epoch) in each epoch."""
     
    u_prompt_gini_reward = """
    Return a CSV that contains the amount of rewards given to each active validating pool 
    (pools that have earned non-zero reward in an epoch) in each epoch."""

    u_prompt_prob = """
    Return the amount of rewards that each staking address receives in epoch 280."""

    u_prompt_num_of_pools1 = """
    Write a single SQL query that returns, for each epoch:

    1.The epoch number.
    2.The total number of validating pools that have received delegation stake in that epoch. (Note: Count should include duplicates, so don't filter out duplicates.)
    3.A list of pool IDs that were active in that epoch (i.e., pools that received a non-zero amount of delegation stake), all combined into a single string in each row.
    The results should have 3 columns:

    epoch_number: the epoch number.
    total_delegations: the total count of validating pools (including duplicates) in that epoch.
    active_pool_ids: a list of pool IDs that were active in that epoch (dont have to be unique).
    """

    u_prompt_num_of_pools2 = """
    Write a single SQL query that returns, for each epoch:

    1.The epoch number.
    2.The total number of validating pools that have received rewards in that epoch. (Note: Count should include duplicates, so don't filter out duplicates.)
    3.A list of pool IDs that were rewarded in that epoch (i.e., pools that received a non-zero amount of reward), all combined into a single string in each row.
    The results should have 3 columns:

    epoch_number: the epoch number.
    total_rewards: the total count of validating pools (including duplicates) in that epoch.
    active_pool_ids: a list of pool IDs that were rewarded in that epoch (dont have to be unique).
    """

    run_sql_agent(u_prompt_num_of_pools2)




# 转化成流式传输csv

# 优化了prompt，尽量使用join而不是left join