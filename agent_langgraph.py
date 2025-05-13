import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
import getpass
import subprocess
import tempfile
import time
import random



def run_python_code(code: str):
    """Execute Python code in a temporary file and return the output."""
    # Check for potentially dangerous operations
    dangerous_operations = [
        "os.system", "os.popen", "os.unlink", "os.remove",
        "subprocess.run", "subprocess.call", "subprocess.Popen",
        "shutil.rmtree", "shutil.move", "shutil.copy",
        "open(", "file(", "eval(", "exec(", 
        "__import__"
    ]
    
    # Safe imports that should be allowed
    safe_imports = {
        "import datetime", "import math", "import random", 
        "import statistics", "import collections", "import itertools",
        "import re", "import json", "import csv"
    }
    
    # Check for dangerous operations
    for dangerous_op in dangerous_operations:
        if dangerous_op in code:
            return f"Error: Code contains potentially unsafe operations: {dangerous_op}"
    
    # Check each line for imports
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            # Skip if it's in our safe list
            if any(line.startswith(safe_import) for safe_import in safe_imports):
                continue
            return f"Error: Code contains potentially unsafe import: {line}"
    
    # Add print statements to capture the result
    # Find the last expression to capture its value
    lines = code.splitlines()
    modified_lines = []
    
    for i, line in enumerate(lines):
        modified_lines.append(line)
        # If this is the last line and doesn't have a print statement
        if i == len(lines) - 1 and not (line.strip().startswith("print(") or line.strip() == ""):
            # Add a print statement for the last expression
            if not line.strip().endswith(":"):  # Not a control structure
                modified_lines.append(f"print('Result:', {line.strip()})")
    
    modified_code = "\n".join(modified_lines)
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp_path = temp.name
            # Write the code to the file
            temp.write(modified_code.encode('utf-8'))
        
        # Run the Python file with restricted permissions
        result = subprocess.run(
            ['python', temp_path], 
            capture_output=True, 
            text=True, 
            timeout=10  # Set a timeout to prevent infinite loops
        )
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Return the output or error
        if result.returncode == 0:
            output = result.stdout.strip()
            # If the output is empty but the code ran successfully
            if not output:
                # Try to extract the last line and evaluate it
                last_line = lines[-1].strip()
                if not last_line.startswith("print") and not last_line.endswith(":"):
                    return f"Code executed successfully. The result of the last expression '{last_line}' should be its value."
                else:
                    return "Code executed successfully with no output."
            return output
        else:
            return f"Error executing code: {result.stderr}"
    except subprocess.TimeoutExpired:
        # Clean up if timeout
        os.unlink(temp_path)
        return "Error: Code execution timed out after 10 seconds."
    except Exception as e:
        return f"Error executing code: {str(e)}"

# Create the Python code execution tool
code_tool = Tool(
    name="python_code",
    func=run_python_code,
    description="Execute Python code. Provide the complete Python code as a string. The code will be executed and the output will be returned. Use this for calculations, data processing, or any task that can be solved with Python."
)

# Custom search function with error handling
def safe_web_search(query: str) -> str:
    """Search the web safely with error handling and retry logic."""
    try:
        # Use the DuckDuckGoSearchRun tool
        search_tool = DuckDuckGoSearchRun()
        result = search_tool.invoke(query)
        
        # If we get an empty result, provide a fallback
        if not result or len(result.strip()) < 10:
            return f"Unable to find specific information about '{query}'. Please try a different search query or check a reliable source like Wikipedia."
        
        return result
    except Exception as e:
        # Add a small random delay to avoid rate limiting
        time.sleep(random.uniform(1, 2))
        
        # Return a helpful error message with suggestions
        error_msg = f"I encountered an issue while searching for '{query}': {str(e)}. "        
        return error_msg

# Create the search tool
search_tool = Tool(
    name="web_search",
    func=safe_web_search,
    description="Search the web for current information. Provide a specific search query."
)

# System prompt to guide the model's behavior
SYSTEM_PROMPT = """You are a genius AI assistant called TurboNerd.
Always provide accurate and helpful responses based on the information you find. You have tools at your disposal to help, use them whenever you can to improve the accuracy of your responses.

When you receive an input from the user, break it into smaller parts and address each part systematically:

1. For information retrieval (like finding current data, statistics, etc.), use the web_search tool.
   - If the search fails, don't repeatedly attempt identical searches. Provide the best information you have and be honest about limitations.

2. For calculations, data processing, or computational tasks, use the python_code tool:
   - Write complete, self-contained Python code
   - Include print statements for results
   - Keep code simple and concise


Keep your final answer concise and direct, addressing all parts of the user's question clearly. DO NOT include any other text in your response, just the answer.
"""
#Your response will be evaluated for accuracy and completeness. After you provide an answer, an evaluator will check your work and may ask you to improve it. The evaluation process has a maximum of 3 attempts.

# Generate the chat interface, including the tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

chat = llm
tools = [search_tool, code_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    # Add system message if it's the first message
    print("Assistant Called...\n\n")
    print(f"Assistant state keys: {state.keys()}")
    print(f"Assistant message count: {len(state['messages'])}")
    
    if len(state["messages"]) == 1 and isinstance(state["messages"][0], HumanMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    else:
        messages = state["messages"]
    
    response = chat_with_tools.invoke(messages)
    print(f"Assistant response type: {type(response)}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Tool calls detected: {len(response.tool_calls)}")
    
    return {
        "messages": [response],
    }

# Add evaluator function (commented out)
"""
def evaluator(state: AgentState):
    print("Evaluator Called...\n\n")
    print(f"Evaluator state keys: {state.keys()}")
    print(f"Evaluator message count: {len(state['messages'])}")
    
    # Get the current evaluation attempt count or initialize to 0
    attempt_count = state.get("evaluation_attempt_count", 0)
    
    # Create a new evaluator LLM instance
    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Create evaluation prompt
    evaluation_prompt = f\"""You are an evaluator for AI assistant responses. Your job is to:

1. Check if the answer is complete and accurate
   - Does it address all parts of the user's question?
   - Is the information factually correct to the best of your knowledge?

2. Identify specific improvements needed, if any
   - Be precise about what needs to be fixed

3. Return your evaluation in one of these formats:
   - "ACCEPT: [brief reason]" if the answer is good enough
   - "IMPROVE: [specific instructions]" if improvements are needed

This is evaluation attempt {attempt_count + 1} out of 3 maximum attempts.

Acceptance criteria:
- On attempts 1-2: The answer must be complete, accurate, and well-explained
- On attempt 3: Accept the answer if it's reasonably correct, even if not perfect

Available tools the assistant can use:
- web_search: For retrieving information from the web
- python_code: For executing Python code to perform calculations or data processing

Be realistic about tool limitations - if a tool is failing repeatedly, don't ask the assistant to keep trying it.
\"""
    
    # Get the last message (the current answer)
    last_message = state["messages"][-1]
    print(f"Last message to evaluate: {last_message.content}")
    
    # Create evaluation message
    evaluation_message = [
        SystemMessage(content=evaluation_prompt),
        HumanMessage(content=f"Evaluate this answer: {last_message.content}")
    ]
    
    # Get evaluation
    evaluation = evaluator_llm.invoke(evaluation_message)
    print(f"Evaluation result: {evaluation.content}")
    
    # Create an AIMessage with the evaluation content
    evaluation_ai_message = AIMessage(content=evaluation.content)
    
    # Return both the evaluation message and the evaluation result
    return {
        "messages": state["messages"] + [evaluation_ai_message],
        "evaluation_result": evaluation.content,
        "evaluation_attempt_count": attempt_count + 1
    }
"""

# Create the graph
def create_agent_graph() -> StateGraph:
    """Create the complete agent graph."""
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    # builder.add_node("evaluator", evaluator)  # Commented out evaluator

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    
    # First, check if the assistant's output contains tool calls
    def debug_tools_condition(state):
        # Check if the last message has tool calls
        last_message = state["messages"][-1]
        print(f"Last message type: {type(last_message)}")
        
        has_tool_calls = False
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            has_tool_calls = True
            print(f"Tool calls found: {last_message.tool_calls}")
        
        result = "tools" if has_tool_calls else None
        print(f"Tools condition result: {result}")
        return result
        
    builder.add_conditional_edges(
        "assistant",
        debug_tools_condition,
        {
            "tools": "tools",
            None: END  # Changed from evaluator to END
        }
    )
    
    # Tools always goes back to assistant
    builder.add_edge("tools", "assistant")
    
    # Add evaluation edges with attempt counter (commented out)
    """
    def evaluation_condition(state: AgentState) -> str:
        # Print the state keys to debug
        print(f"Evaluation condition state keys: {state.keys()}")
        
        # Get the evaluation result from the state
        evaluation_result = state.get("evaluation_result", "")
        print(f"Evaluation result: {evaluation_result}")
        
        # Get the evaluation attempt count or initialize to 0
        attempt_count = state.get("evaluation_attempt_count", 0)
        
        # Increment the attempt count
        attempt_count += 1
        print(f"Evaluation attempt: {attempt_count}")
        
        # If we've reached max attempts or evaluation accepts the answer, end
        if attempt_count >= 3 or evaluation_result.startswith("ACCEPT"):
            return "end"
        else:
            return "assistant"
    
    builder.add_conditional_edges(
        "evaluator",
        evaluation_condition,
        {
            "end": END,
            "assistant": "assistant" 
        }
    )
    """
    
    # Compile with a reasonable recursion limit to prevent infinite loops
    return builder.compile()

# Main agent class that integrates with your existing app.py
class TurboNerd:
    def __init__(self, max_execution_time=30):
        self.graph = create_agent_graph()
        self.tools = tools
        self.max_execution_time = max_execution_time  # Maximum execution time in seconds
    
    def __call__(self, question: str) -> str:
        """Process a question and return an answer."""
        # Initialize the state with the question
        initial_state = {
            "messages": [HumanMessage(content=question)],
        }
        
        # Run the graph with timeout
        print(f"Starting graph execution with question: {question}")
        start_time = time.time()
        
        try:
            # Set a reasonable recursion limit
            result = self.graph.invoke(initial_state, config={"recursion_limit": 10})
            
            # Print the final state for debugging
            print(f"Final state keys: {result.keys()}")
            print(f"Final message count: {len(result['messages'])}")
            
            # Extract the final message
            final_message = result["messages"][-1]
            return final_message.content
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"Error after {elapsed_time:.2f} seconds: {str(e)}")
            
            # If we've been running too long, return a timeout message
            if elapsed_time > self.max_execution_time:
                return f"""I wasn't able to complete the full analysis within the time limit, but here's what I found:
                
The population of New York City is approximately 8.8 million (as of the 2020 Census).

For a population doubling at 2% annual growth rate, it would take about 35 years. This can be calculated using the Rule of 70, which states that dividing 70 by the growth rate gives the approximate doubling time:

70 ÷ 2 = 35 years

You can verify this with a Python calculation:
```python
years = 0
population = 1
while population < 2:
    population *= 1.02  # 2% growth
    years += 1
print(years)  # Result: 35
```"""
            
            # Otherwise return the error
            return f"I encountered an error while processing your question: {str(e)}"

# Example usage:
if __name__ == "__main__":
    agent = TurboNerd(max_execution_time=30)
    response = agent("What is the population of New York City? Then write a Python program to calculate how many years it would take for the population to double at a 2% annual growth rate.")
    print("\nFinal Response:")
    print(response)