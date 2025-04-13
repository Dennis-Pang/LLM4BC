from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from typing import Any
from langgraph.prebuilt import ToolNode
import os
from langgraph.graph.message import AnyMessage, add_messages
from typing import Any, Annotated, Literal
from typing_extensions import TypedDict
import subprocess

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def create_new_state() -> State:
    return {
        "messages": []
    }

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def run_python_file_subprocess(file_path):
    try:
        result = subprocess.run(['python', file_path], capture_output=True, text=True)
        print(f"Output from {file_path}:\n{result.stdout}")
        if result.stderr:
            print(f"Error occurred:\n{result.stderr}")
    except Exception as e:
        print(f"Error occurred: {e}")      

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def get_case_number(directory: str, prefix: str = "case") -> int:
    """
    Get the next case number by scanning the directory for existing case folders.
    The case folders follow the pattern 'case<number>'.
    """
    existing_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    case_number = [
        int(d.replace(prefix, ''))
        for d in existing_dirs
        if d.startswith(prefix) and d[len(prefix):].isdigit()
    ]
    return max(case_number)

def get_next_case_number(directory: str, prefix: str = "case") -> int:
    """
    Get the next case number by scanning the directory for existing case folders.
    The case folders follow the pattern 'case<number>'.
    """
    existing_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    case_number = [
        int(d.replace(prefix, ''))
        for d in existing_dirs
        if d.startswith(prefix) and d[len(prefix):].isdigit()
    ]
    
    if case_number:
        return max(case_number) + 1
    else:
        return 1
    
