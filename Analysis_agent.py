from dotenv import load_dotenv
import os
from typing import Literal
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from langchain_core.runnables.graph import MermaidDrawMethod
from general_functions import create_tool_node_with_fallback, get_case_number, State, create_new_state, run_python_file_subprocess


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# results_dir = 'Project_Pythonfile/results'

project_root = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(project_root, 'results')

case_path = f"case{get_case_number(results_dir)}"

project_root = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(project_root, 'results')

file_path = os.path.join(results_dir, case_path)
Input_dir=Output_dir=file_path

# 生成绝对路径

Input_file = "query_result.csv"

# Define the tool to read CSV and return schema
@tool
def understand_csv_schema():
    """
    Reads the 'tables_schema.txt' file and returns its content.
    """
    print('"understand_csv_schema" started (1)')
    info = {}
    file_path = os.path.join(Input_dir, Input_file)
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        schema = df.dtypes
        
        info["csv_schema"] = schema
    except FileNotFoundError:
        raise FileNotFoundError("CSV file not found, please check the function 'get_tables_schema' of the SQL agent.")
    try:
        with open(os.path.join(Input_dir,'tables_schema.txt'), 'r', encoding='utf-8') as output_file:
            info["table_schema"] = output_file.read()
    except FileNotFoundError:
        raise FileNotFoundError("Tables schema file not found, please check the function 'get_tables_schema' of the SQL agent.")
    return info

# Define the system prompt for information checking
information_check_system =  """
As a data processing agent, your task is to evaluate user requests and determine if data processing is needed based on the schema of the data.

** Only pay attention to the data processing demand! 
** Don't do any visualization!

Follow these steps:

0. **First, use understand_csv_schema**:
   - Read the description of the schema of the CSV file and the schema of the tables used for the csv.

1. **Read the csv file**:
   - The input file's name is {Input_file} in {Input_dir}! There's only one file, you should read all attributes from {Input_file}.
   - If the user's request includes operations such as calculating the average, median, filtering, sorting, or any other data operations, return the necessary Python code that can be directly executed with `exec(1_processing_instructions)`.
   - If no data processing is needed, return "end" and pass the original data without any modification.

2. **Ensure schema consistency**:
   - When generating the Python code, ensure that the column names used in the code exactly match the column names in the CSV schema, including case sensitivity and spelling. You must not assume any changes to the column names provided by the user. All column names should be programmatically verified against the schema.

3. **If data processing is needed**:
   - Only return the necessary Python code to process the data (e.g., filtering, aggregation, statistical calculations) in a format that can be executed directly in `exec()`.
   - The code should modify the DataFrame `df` based on the schema and save it back to the CSV file. Any intermediate variables should be included in the provided Python code.
   - Ensure the code is robust and handles edge cases such as missing values or incorrect column names.
   - Save the code to the {Input_dir} with name "processing_code.py"

4. **File Saving**:
   - After processing, ensure the modified CSV file is saved to the {Input_dir} directory and name the modified CSV file as "1_processed_data"; 

### Response format:
- If processing is required: 
    Call "save_and_run" and pass the necessary Python code to be executed.

- If no processing is required: Call "end" without any Python code.
""".format(Input_file=Input_file, Input_dir=Input_dir)



@tool
def save_and_run(code:str) -> str:
    """
       read, process instructions, save
    """
    print("'save_and_run' started (3)")
    directory = Output_dir
    csv_file = None
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            csv_file = os.path.join(directory, file_name)
            break  
    if csv_file:
        df = pd.read_csv(csv_file)
    else:
        raise FileNotFoundError("CSV file not found in the specified directory.")
    
    try:
        file_path = os.path.join(Input_dir, "1_processing_instructions.py")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(code)
        run_python_file_subprocess(file_path)    
        print('"process and save" succeeded (4)')
        return END
        
    except Exception as e:
        raise  ValueError(f"Error processing data: {str(e)}")
    
# Define the chat prompt template for information check
information_check_prompt = ChatPromptTemplate.from_messages(
    [("system", information_check_system), ("user", "{messages}")]
)

# Combine the prompt template with the OpenAI model
information_check = information_check_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [save_and_run]
)

def information_check_node(state: State) -> dict[str, list[AIMessage]]:
    print('"information_check" started (2)')
    response = information_check.invoke(state)
    return {"messages": [response]}

# Define a new graph
state = create_new_state()
workflow = StateGraph(State)

def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "understand_csv_schema",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("understand_csv_schema", create_tool_node_with_fallback([understand_csv_schema]))
workflow.add_node(
    "save_and_run", create_tool_node_with_fallback([save_and_run])
)
workflow.add_node("Analysis_node", information_check_node)

workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call","understand_csv_schema")
workflow.add_edge("understand_csv_schema","Analysis_node")

def should_continue_ana(state: State) -> Literal["save_and_run", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        for tc in last_message.tool_calls:
            if tc["name"] == "save_and_run":
                return "save_and_run"
    return END

workflow.add_conditional_edges(
    "Analysis_node",
    should_continue_ana
)

def should_continue_save_and_run(state: State) -> Literal['Analysis_node', END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content.startswith("Error"):
        return "Analysis_node"
    return END 

workflow.add_conditional_edges(
    "save_and_run",
    should_continue_save_and_run
)
app = workflow.compile()

'''
# get image
img_data = app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)

# save image
with open("./Project_Pythonfile/Agent_pipeline_charts/analysis_agent.png", "wb") as file:
    file.write(img_data)

'''

def run_ana_agent(input_data):
    final_state = app.invoke(
        {"messages": [("user", input_data)]}
    )
    return final_state


if __name__ == "__main__":
    u_prompt_median_regression = """
Use logarithmic binning with 100 bins for the stakes delegated to each pool, ensuring both stake and reward values are non-negative (can be 0, but must not be negative).
For each bin:
1. Calculate the median reward that the pools in that bin received.
2. Save the central stake value (the midpoint of the bin) and the corresponding median reward.

Only return the following 3 columns in the output:
0. The bin number (starting from 1)
1. The central stake value for each bin (can be 0). Divide number by 10^6 into ADA format.
2. The median reward for each bin. Divide number by 10^6 into ADA format.

Ensure that:
- Bins are correctly defined with logarithmic scaling.
- Handle empty bins by returning a 0 for median reward.
- Avoid index out-of-bounds errors when binning or retrieving values.
"""
    u_prompt_gini_stake = """
Calculate the "Gini index" of stakes delegated to the pools in each epoch. 
Make sure to convert any Pandas Series to a numpy array before performing numpy operations.
"""
    u_prompt_gini_reward = """
Calculate the "Gini index" of rewards given to the pools in each epoch.
Make sure to convert any Pandas Series to a numpy array before performing numpy operations.
"""
    u_prompt_prob = """
Use logarithmic binning with 100 bins for the 'total_rewards' values, ensuring that only positive reward values are considered.
Calculate the histogram (or probability density) of the rewards for the staking addresses within each bin.
For each bin, calculate the central stake value (in ADA, by dividing the central value by 10^6).
Store both the positive probability density(multiply the probability density by 10^6) and the central stake value (in ADA format) in the output.
"""
    u_prompt_num_of_pools = """
Calculate the "cumulative" number of pools (don't store the uniqle ids).Only return 2 columns: epoch number and "cumulative" number of pools

For calculating the cumulative values, you should consider that the pool_ids repeated in different epochs should be counted only once.
Therefore, for each epoch "i", you need to look into the list of all pools in that epoch so that you can find the number of unique pools from epoch 1 to epoch "i".
For example, if a pool_id has been active in epochs 280, 300, and 320, it should be counted as 1 pool_id in the "cumulative" number of pools in epoch 320
 """

    run_ana_agent(u_prompt_num_of_pools)

