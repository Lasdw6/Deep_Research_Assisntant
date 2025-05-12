import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

def calculator(operation: str, num1: int, num2: int):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2
    return input

calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description="Use this tool to perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers"
)

# Initialize the web search tool
search_tool = DuckDuckGoSearchRun()

# System prompt to guide the model's behavior
SYSTEM_PROMPT = """You are a genuis AI assistant called TurboNerd.
Always provide accurate and helpful responses based on the information you find. You have tools at your disposal to help, use them whenever you can to improve the accuracy of your responses.
When you recieve an input from the user, first you will break the input into smaller parts. Then you will one by one use the tools to answer the question. The final should be as short as possible, directly answering the question.
"""

# Generate the chat interface, including the tools
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

chat = llm
tools = [search_tool, calculator_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    # Add system message if it's the first message
    print("Assistant called... \n")
    if len(state["messages"]) == 1 and isinstance(state["messages"][0], HumanMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    else:
        messages = state["messages"]
    
    return {
        "messages": [chat_with_tools.invoke(messages)],
    }

# Create the graph
def create_agent_graph() -> StateGraph:
    """Create the complete agent graph."""
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    
    return builder.compile()

# Main agent class that integrates with your existing app.py
class TurboNerd:
    def __init__(self):
        self.graph = create_agent_graph()
        self.tools = tools
    
    def __call__(self, question: str) -> str:
        """Process a question and return an answer."""
        # Initialize the state with the question
        initial_state = {
            "messages": [HumanMessage(content=question)],
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract the final message
        final_message = result["messages"][-1]
        return final_message.content

# Example usage:
if __name__ == "__main__":
    agent = TurboNerd()
    response = agent("What is the time in Tokyo now?")
    print(response)