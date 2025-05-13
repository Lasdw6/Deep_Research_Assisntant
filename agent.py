import os
from typing import TypedDict, Annotated, Dict, Any, Optional, Union, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
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
import json
import re
import requests
from urllib.parse import quote
import sys

from apify_client import ApifyClient




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

# Apify-based search function
def apify_google_search(query: str, limit: int = 10) -> str:
    """
    Use Apify's Google Search Results Scraper to get search results
    
    Args:
        query: The search query string
        limit: Number of results to return (10, 20, 30, 40, 50, 100)
        
    Returns:
        Formatted search results as a string
    """
    # You would need to provide a valid Apify API token
    # You can get one by signing up at https://apify.com/
    # Replace this with your actual Apify API token or set as environment variable
    APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
    
    if not APIFY_API_TOKEN:
        print("No Apify API token found. Using fallback search method.")
        return fallback_search(query)
    
    try:
        # Initialize the ApifyClient with API token
        client = ApifyClient(APIFY_API_TOKEN)
        
        # Prepare the Actor input - convert limit to string as required by the API
        run_input = {
            "keyword": query,
            "limit": str(limit),  # Convert to string as required by the API
            "country": "US"
        }
        
        # The Actor ID for the Google Search Results Scraper
        ACTOR_ID = "563JCPLOqM1kMmbbP"
        
        print(f"Starting Apify search for: '{query}'")
        
        # Run the Actor and wait for it to finish (with timeout)
        run = client.actor(ACTOR_ID).call(run_input=run_input, timeout_secs=60)
        
        if not run or not run.get("defaultDatasetId"):
            print("Failed to get results from Apify actor")
            return fallback_search(query)
            
        # Fetch Actor results from the run's dataset
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)
        
        # Format and return the results
        return format_search_results(results, query)
        
    except Exception as e:
        print(f"Error using Apify: {str(e)}")
        return fallback_search(query)

def format_search_results(results: List[Dict], query: str) -> str:
    """Format the search results into a readable string"""
    if not results or len(results) == 0:
        return f"No results found for query: {query}"
    
    print(f"Raw search results: {str(results)[:1000]}...")
    
    # Extract search results from the Apify output
    formatted_results = f"Search results for '{query}':\n\n"
    
    # Check if results is a list of dictionaries or a dictionary with nested results
    if isinstance(results, dict) and "results" in results:
        items = results["results"]
    elif isinstance(results, list):
        items = results
    else:
        return f"Unable to process results for query: {query}"
    
    # Handle different Apify result formats
    if len(items) > 0:
        # Check the structure of the first item to determine format
        first_item = items[0]
        
        # If item has 'organicResults', this is the format from some Apify actors
        if isinstance(first_item, dict) and "organicResults" in first_item:
            organic_results = first_item.get("organicResults", [])
            for i, result in enumerate(organic_results[:10], 1):
                if "title" in result and "url" in result:
                    formatted_results += f"{i}. {result['title']}\n"
                    formatted_results += f"   URL: {result['url']}\n"
                    if "snippet" in result:
                        formatted_results += f"   {result['snippet']}\n"
                    formatted_results += "\n"
        else:
            # Standard format with title/url/description
            for i, result in enumerate(items[:10], 1):
                if "title" in result and "url" in result:
                    formatted_results += f"{i}. {result['title']}\n"
                    formatted_results += f"   URL: {result['url']}\n"
                    if "description" in result:
                        formatted_results += f"   {result['description']}\n"
                    elif "snippet" in result:
                        formatted_results += f"   {result['snippet']}\n"
                    formatted_results += "\n"
    
    return str(items)

def fallback_search(query: str) -> str:
    """Fallback search method using DuckDuckGo when Apify is not available"""
    try:
        search_tool = DuckDuckGoSearchRun()
        result = search_tool.invoke(query)
        return "Observation: " + result
    except Exception as e:
        return f"Search error: {str(e)}. Please try a different query or method."

# Custom search function with improved error handling
def safe_web_search(query: str) -> str:
    """Search the web safely with error handling and retry logic."""
    if not query:
        return "Error: No search query provided. Please specify what you want to search for."
    
    # Try using Apify first, if it fails it will use the fallback
    return "Observation: " + apify_google_search(query)
    
    # The code below is kept for reference but won't be executed
    max_retries = 3
    backoff_factor = 1.5
    
    for attempt in range(max_retries):
        try:
            # Use the DuckDuckGoSearchRun tool
            search_tool = DuckDuckGoSearchRun()
            result = search_tool.invoke(query)
            
            # If we get an empty result, provide a helpful message
            if not result or len(result.strip()) < 10:
                return f"The search for '{query}' did not return any useful results. Please try a more specific query or a different search engine."
            
            return "Observation: " + result
            
        except Exception as e:
            # If we're being rate limited
            if "Ratelimit" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"Rate limited, waiting {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # On last attempt, return a helpful error
                    error_msg = f"I'm currently unable to search for '{query}' due to service rate limits. "
                    return error_msg
            else:
                # For other types of errors
                return f"Error while searching for '{query}': {str(e)}"
            
    return f"Failed to search for '{query}' after multiple attempts due to rate limiting."

# System prompt to guide the model's behavior
SYSTEM_PROMPT = """Answer the following questions as best you can. DO NOT rely on your internal knowledge unless web searches are rate-limited or you're specifically instructed to. You have access to the following tools:

web_search: Search the web for current information. Provide a specific search query.
python_code: Execute Python code. Provide the complete Python code as a string. Use this tool to calculate math problems.

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
web_search: Search the web for current information, args: {"query": {"type": "string"}}
python_code: Execute Python code, args: {"code": {"type": "string"}}

IMPORTANT: Make sure your JSON is properly formatted with double quotes around keys and string values.

example use:

```json
{
  "action": "web_search",
  "action_input": {"query": "population of New York City"}
}
```

ALWAYS follow this specific format for your responses. Your entire response will follow this pattern:

Question: [the user's question]

Thought: [your reasoning about what to do next]

Action: 
```json
{
  "action": "[tool_name]",
  "action_input": {"[parameter_name]": "[parameter_value]"}
}
```

Observation: [the result from the tool will appear here]

Thought: [your reasoning after seeing the observation]

Action: 
```json
{
  "action": "[tool_name]",
  "action_input": {"[parameter_name]": "[parameter_value]"}
}
```

Observation: [another tool result will appear here]

IMPORTANT: You MUST strictly follow the ReAct pattern (Reasoning, Action, Observation):
1. First reason about the problem in the "Thought" section
2. Then decide what action to take in the "Action" section (using the tools)
3. Wait for an observation from the tool
4. Based on the observation, continue with another thought
5. This cycle repeats until you have enough information to provide a final answer

... (this Thought/Action/Observation cycle can repeat as needed) ...

Thought: I now know the final answer

Final Answer: Directly answer the question in the shortest possible way. For example, if the question is "What is the capital of France?", the answer should be "Paris" without any additional text. If the question is "What is the population of New York City?", the answer should be "8.4 million" without any additional text.

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer."""
#Your response will be evaluated for accuracy and completeness. After you provide an answer, an evaluator will check your work and may ask you to improve it. The evaluation process has a maximum of 3 attempts.

# Generate the chat interface, including the tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

chat = llm
# Tools are defined but not bound to the LLM here
tools_config = [
    {
        "name": "web_search",
        "description": "Search the web for current information. Provide a specific search query in the format: {\"query\": \"your search query here\"}",
        "func": safe_web_search
    },
    {
        "name": "python_code", 
        "description": "Execute Python code. Provide the complete Python code as a string in the format: {\"code\": \"your python code here\"}",
        "func": run_python_code
    }
]

# Instead of binding tools, we'll handle the JSON parsing ourselves
# chat_with_tools = chat.bind_tools([Tool(**tool) for tool in tools_config])
chat_with_tools = chat

# Generate the AgentState and Agent graph
class ActionInput(TypedDict, total=False):
    query: Optional[str]
    code: Optional[str]

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    current_tool: Optional[str]
    action_input: Optional[ActionInput]

def assistant(state: AgentState) -> Dict[str, Any]:
    """Assistant node that processes messages and decides on next action."""
    print("Assistant Called...\n\n")
    
    # Always include the system message at the beginning of the messages list
    # This ensures the LLM follows the correct ReAct pattern in every call
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # Get user messages from state, but leave out any existing system messages
    user_messages = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
    
    # Combine system message with user messages
    messages = [system_msg] + user_messages
    
    # Print the full context of messages being sent to the LLM
    print("\n=== INPUT TO LLM ===")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content_preview = msg.content + "..." if len(msg.content) > 150 else msg.content
        print(f"Message {i} ({msg_type}): {content_preview}")
    print("=== END INPUT ===\n")
    
    # Get response from the assistant
    response = chat_with_tools.invoke(messages, stop=["Observation:"])
    print(f"Assistant response type: {type(response)}")
    print(f"Response content: {response.content}...")
    
    # Extract the action JSON from the response text
    action_json = extract_json_from_text(response.content)
    print(f"Extracted action JSON: {action_json}")
    
    # Create a copy of the assistant's response to add to the message history
    assistant_message = AIMessage(content=response.content)
    
    if action_json and "action" in action_json and "action_input" in action_json:
        tool_name = action_json["action"]
        tool_input = action_json["action_input"]
        print(f"Extracted tool: {tool_name}")
        print(f"Tool input: {tool_input}")
        
        # Create a tool call ID for the ToolMessage
        tool_call_id = f"call_{random.randint(1000000, 9999999)}"
        
        # Create state update with the assistant's response included
        state_update = {
            "messages": state["messages"] + [assistant_message],  # Add full assistant response to history
            "current_tool": tool_name,
            "tool_call_id": tool_call_id
        }
        
        # Add action_input to state
        if isinstance(tool_input, dict):
            state_update["action_input"] = tool_input
        
        return state_update
    
    # No tool calls or end of chain indicated by "Final Answer"
    if "Final Answer:" in response.content:
        print("Final answer detected")
    
    return {
        "messages": state["messages"] + [assistant_message],  # Add full assistant response to history
        "current_tool": None,
        "action_input": None
    }

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, handling markdown code blocks."""
    try:
        print(f"Attempting to extract JSON from text: {text[:100]}...")
        
        # Look for markdown code blocks - the most common pattern
        if "```" in text:
            print("Found markdown code block")
            # Find all code blocks
            blocks = []
            in_block = False
            start_pos = 0
            
            for i, line in enumerate(text.split('\n')):
                if "```" in line and not in_block:
                    in_block = True
                    start_pos = i + 1  # Start on the next line
                elif "```" in line and in_block:
                    in_block = False
                    # Extract the block content
                    block_content = '\n'.join(text.split('\n')[start_pos:i])
                    blocks.append(block_content)
            
            # Try to parse each block as JSON
            for block in blocks:
                block = block.strip()
                print(f"Trying to parse block: {block[:100]}...")
                try:
                    # Clean the block - sometimes there might be a language identifier
                    if block.startswith("json"):
                        block = block[4:].strip()
                    
                    return json.loads(block)
                except:
                    continue
        
        # Look for JSON-like patterns in the text
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        # If we're here, we couldn't find a valid JSON object
        print("Could not extract valid JSON from text")
        return None
        
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Node that executes the web search tool."""
    print("Web Search Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Web search action_input: {action_input}")
    
    # Try different ways to extract the query
    query = ""
    if isinstance(action_input, dict):
        query = action_input.get("query", "")
    elif isinstance(action_input, str):
        query = action_input
    
    print(f"Searching for: '{query}'")
    
    # Call the search function with retry logic
    result = safe_web_search(query)
    print(f"Search result: {result}")  # Print the full result for debugging
    
    # Check if we hit rate limits and add a helpful note
    if "rate limit" in result.lower() or "ratelimit" in result.lower():
        result += "\n\nNote: You can use your internal knowledge to provide a response since the search is rate limited."
    
    # Format the observation to continue the ReAct cycle
    # Don't include "Observation:" as the assistant is stopped at this token
    observation = result
    
    # Create a tool message with the result
    tool_message = AIMessage(
        content=f"Observation: {observation}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    print(tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def python_code_node(state: AgentState) -> Dict[str, Any]:
    """Node that executes Python code."""
    print("Python Code Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Python code action_input: {action_input}")
    
    # Try different ways to extract the code
    code = ""
    if isinstance(action_input, dict):
        code = action_input.get("code", "")
    elif isinstance(action_input, str):
        code = action_input
    
    print(f"Executing code: '{code[:100]}...'")
    
    # Safety check - don't run empty code
    if not code:
        result = "Error: No Python code provided. Please provide valid Python code to execute."
    else:
        # Call the code execution function
        result = run_python_code(code)
    
    print(f"Code execution result: {result[:100]}...")  # Print first 100 chars
    
    # Format the observation to continue the ReAct cycle
    # Create a tool message with the result
    tool_message = AIMessage(
        content=f"Observation: {result}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    print(tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

# Router function to direct to the correct tool
def router(state: AgentState) -> str:
    """Route to the appropriate tool based on the current_tool field."""
    tool = state.get("current_tool")
    action_input = state.get("action_input")
    print(f"Routing to: {tool}")
    print(f"Router received action_input: {action_input}")
    
    if tool == "web_search":
        return "web_search"
    elif tool == "python_code":
        return "python_code"
    else:
        return "end"

# Create the graph
def create_agent_graph() -> StateGraph:
    """Create the complete agent graph with individual tool nodes."""
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("web_search", web_search_node)
    builder.add_node("python_code", python_code_node)

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    
    # Debug the state passing
    def debug_state(state):
        print("\n=== DEBUG STATE ===")
        print(f"State keys: {state.keys()}")
        print(f"Current tool: {state.get('current_tool')}")
        print(f"Action input: {state.get('action_input')}")
        print("=== END DEBUG ===\n")
        return state
    
    # Add a checkpoint between nodes to track state
    builder.add_node("debug", debug_state)
    
    # Conditional edge from assistant to debug
    builder.add_edge("assistant", "debug")
    
    # Conditional edge from debug to tools or end
    builder.add_conditional_edges(
        "debug",
        router,
        {
            "web_search": "web_search",
            "python_code": "python_code",
            "end": END
        }
    )
    
    # Tools always go back to assistant
    builder.add_edge("web_search", "assistant")
    builder.add_edge("python_code", "assistant")
    
    # Compile with a reasonable recursion limit to prevent infinite loops
    return builder.compile()

# Main agent class that integrates with your existing app.py
class TurboNerd:
    def __init__(self, max_execution_time=60, apify_api_token=None):
        self.graph = create_agent_graph()
        self.tools = tools_config
        self.max_execution_time = max_execution_time  # Maximum execution time in seconds
        
        # Set Apify API token if provided
        if apify_api_token:
            os.environ["APIFY_API_TOKEN"] = apify_api_token
            print("Apify API token set successfully")
    
    def __call__(self, question: str) -> str:
        """Process a question and return an answer."""
        # Initialize the state with the question
        initial_state = {
            "messages": [HumanMessage(content=f"Question: {question}")],
            "current_tool": None,
            "action_input": None
        }
        
        # Run the graph with timeout
        print(f"Starting graph execution with question: {question}")
        start_time = time.time()
        
        try:
            # Set a reasonable recursion limit
            result = self.graph.invoke(initial_state, config={"recursion_limit": 15})
            
            # Print the final state for debugging
            print(f"Final state keys: {result.keys()}")
            print(f"Final message count: {len(result['messages'])}")
            
            # Extract the final message and return just the final answer
            final_message = result["messages"][-1].content
            print("Final message: ", final_message)
            # Extract just the final answer part
            if "Final Answer:" in final_message:
                final_answer = final_message.split("Final Answer:")[1].strip()
                return final_answer
            
            return final_message
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            # Otherwise return the error
            return f"I encountered an error while processing your question: {str(e)}"

# Example usage:
if __name__ == "__main__":
    agent = TurboNerd(max_execution_time=60)
    response = agent("How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.")
    print("\nFinal Response:")
    print(response)