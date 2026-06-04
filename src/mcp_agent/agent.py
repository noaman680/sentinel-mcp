from langgraph.graph import StateGraph

from state import AgentState
from nodes import process_message

builder = StateGraph(AgentState)

builder.add_node("process_message", process_message)

builder.set_entry_point("process_message")
builder.set_finish_point("process_message")

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke(
        {
            "input": "Hello LangGraph"
        }
    )

    print(result)