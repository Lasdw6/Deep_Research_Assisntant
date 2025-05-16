import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Dict, Any, Optional, Union, List
from pathlib import Path
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
import tempfile
import random
import json
import requests
from urllib.parse import quote, urlparse
from bs4 import BeautifulSoup
import html2text
import pandas as pd
from tabulate import tabulate
import base64

# Import all tool functions from tools.py
from tools import (
    tools_config,
    run_python_code,
    scrape_webpage,
    wikipedia_search,
    tavily_search,
    arxiv_search,
    supabase_operation,
    excel_to_text,
    save_attachment_to_tempfile,
    process_youtube_video,
    transcribe_audio,
    extract_python_code_from_complex_input,
    process_image,
    read_file,
    process_online_document
)

load_dotenv()

# Remove the following functions from agent.py since they're now imported from tools.py:
# - run_python_code (lines ~28-175)
# - scrape_webpage (lines ~177-310) 
# - wikipedia_search (lines ~345-405)
# - tavily_search (lines ~407-470)
# - arxiv_search (lines ~472-535)
# - supabase_operation (lines ~537-620)
# - excel_to_text (lines ~622-690)
# - save_attachment_to_tempfile (lines ~1680-1706)

# Also remove the tools_config definition (lines ~795-870) since it's imported from tools.py

# The rest of the file remains the same...

# System prompt to guide the model's behavior
#web_search: Search the google search engine when Tavily Search and Wikipedia Search do not return a result. Provide a specific search query.
#webpage_scrape: Scrape content from a specific webpage URL when Tavily Search and Wikipedia Search do not return a result. Provide a valid URL to extract information from a particular web page.
#Give preference to using Tavily Search and Wikipedia Search before using web_search or webpage_scrape. When Web_search does not return a result, use Tavily Search.

SYSTEM_PROMPT = """Answer the following questions as best you can. DO NOT rely on your internal knowledge unless the tools fail to provide a result:

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
python_code: Execute Python code. Use this tool to calculate math problems. make sure to use prints to be able to view the final result. args: {"code": {"type": "string"}}
wikipedia_search: Search Wikipedia for information about a specific topic. Optionally specify the number of results to return, args: {"query": {"type": "string"}, "num_results": {"type": "integer", "optional": true}}
tavily_search: Search the web using Tavily. Optionally specify search_depth as 'basic' or 'comprehensive', args: {"query": {"type": "string"}, "search_depth": {"type": "string", "optional": true}}
arxiv_search: Search ArXiv for publications,news and other resources. Optionally specify max_results to control the number of papers returned, args: {"query": {"type": "string"}, "max_results": {"type": "integer", "optional": true}}
webpage_scrape: Scrape a specific webpage, args: {"url": {"type": "string"}}
supabase_operation: Perform database operations, args: {"operation_type": {"type": "string"}, "table": {"type": "string"}, "data": {"type": "object", "optional": true}, "filters": {"type": "object", "optional": true}}
excel_to_text: Convert Excel to Markdown table with attachment, args: {"excel_path": {"type": "string"}, "file_content": {"type": "string"}, "sheet_name": {"type": "string", "optional": true}}
process_youtube_video: Extract and analyze YouTube video content by providing the video URL. Returns video metadata and transcript, args: {"url": {"type": "string"}, "summarize": {"type": "boolean", "optional": true}}
transcribe_audio: Transcribe audio files using OpenAI Whisper, args: {"audio_path": {"type": "string"}, "file_content": {"type": "string", "optional": true}, "language": {"type": "string", "optional": true}}
process_image: Process and analyze image files, args: {"image_path": {"type": "string"}, "image_url": {"type": "string", "optional": true}, "file_content": {"type": "string", "optional": true}, "analyze_content": {"type": "boolean", "optional": true}}
read_file: Read and display the contents of a text file, args: {"file_path": {"type": "string"}, "file_content": {"type": "string", "optional": true}, "line_start": {"type": "integer", "optional": true}, "line_end": {"type": "integer", "optional": true}}

If you get stuck, try using another tool. For example if you are unable to find relevant information from the tavily_search tool, try using the wikipedia_search tool and vice versa.
IMPORTANT: Make sure your JSON is properly formatted with double quotes around keys and string values.

Example use for tools:

```json
{
  "action": "tavily_search",
  "action_input": {"query": "What is the capital of France?", "search_depth": "basic"}
}
```
or
```json
{
  "action": "process_youtube_video",
  "action_input": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "summarize": true}
}
```
or
```json
{
  "action": "process_image",
  "action_input": {"image_path": "example.jpg", "analyze_content": true}
}
```

ALWAYS follow this specific format for your responses. Your entire response will follow this pattern:
Question: [the user's question]
Thought: [your reasoning about what to do next, break it down into smaller steps and clearly state your thoughts]
Action: 
```json
{
  "action": "[tool_name]",
  "action_input": {"[parameter_name]": "[parameter_value]"}
}
```
Observation: [the result from the tool will appear here]
Thought: [your reasoning after seeing the observation, break it down into smaller steps]
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

NEVER fake or simulate tool output yourself. If you are unable to make progreess in a certain way, try a different tool or a different approach.

... (this Thought/Action/Observation cycle can repeat as needed) ...
Thought: I now know the final answer
Final Answer: YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string. For one word answers, start with the word with a capital letter.
Make sure to follow any formatting instructions given by the user.
Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer."""

# Generate the chat interface, including the tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

chat = llm
# Tools are defined but not bound to the LLM here
tools_config = [
    # {
    #     "name": "web_search",
    #     "description": "Search the web for current information. Provide a specific search query in the format: {\"query\": \"your search query here\"}",
    #     "func": safe_web_search
    # },
    {
        "name": "python_code", 
        "description": "Execute Python code. Provide the complete Python code as a string in the format: {\"code\": \"your python code here\"}",
        "func": run_python_code
    },
    # {
    #     "name": "webpage_scrape",
    #     "description": "Scrape content from a specific webpage URL. Provide a valid URL in the format: {\"url\": \"https://example.com\"}",
    #     "func": scrape_webpage
    # },
    {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for information about a specific topic. Provide a query in the format: {\"query\": \"your topic\", \"num_results\": 3}",
        "func": wikipedia_search
    },
    {
        "name": "tavily_search",
        "description": "Search the web using Tavily for more comprehensive results. Provide a query in the format: {\"query\": \"your search query\", \"search_depth\": \"basic\"}",
        "func": tavily_search
    },
    {
        "name": "arxiv_search",
        "description": "Search ArXiv for scientific papers. Provide a query in the format: {\"query\": \"your research topic\", \"max_results\": 5}",
        "func": arxiv_search
    },
    {
        "name": "supabase_operation",
        "description": "Perform database operations on Supabase (insert, select, update, delete). Provide operation_type, table name, and optional data/filters. ",
        "func": supabase_operation
    },
    {
        "name": "excel_to_text",
        "description": "Read an Excel file and return a Markdown table. You can provide either the path to an Excel file or use a file attachment. For attachments, provide a base64-encoded string of the file content and a filename.",
        "func": excel_to_text
    },
    {
        "name": "process_youtube_video",
        "description": "Extract and analyze YouTube video content by providing the video URL. Returns video metadata and transcript.",
        "func": process_youtube_video
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribe audio files using OpenAI Whisper. You can provide either a file path or use a file attachment. For attachments, provide base64-encoded content. Optionally specify language for better accuracy.",
        "func": transcribe_audio
    }
]

# Instead of binding tools, we'll handle the JSON parsing ourselves
# chat_with_tools = chat.bind_tools([Tool(**tool) for tool in tools_config])
chat_with_tools = chat

# Generate the AgentState and Agent graph
class ActionInput(TypedDict, total=False):
    query: Optional[str]
    code: Optional[str]
    url: Optional[str]
    image_url: Optional[str]
    num_results: Optional[int]
    search_depth: Optional[str]
    max_results: Optional[int]

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    current_tool: Optional[str]
    action_input: Optional[ActionInput]
    iteration_count: int  # Added to track iterations
    attachments: Dict[str, str]  # Added to store file attachments (filename -> base64 content)
    # tool_call_id: Optional[str] # Ensure this is present if used by your graph logic for tools

# Add prune_messages_for_llm function
def prune_messages_for_llm(
    full_history: List[AnyMessage], 
    num_recent_to_keep: int = 6  # Keeps roughly 2-3 ReAct turns (Thought/Action, Observation)
) -> List[AnyMessage]:
    """
    Prunes the message history for the LLM call.
    This function expects a 'core' history (messages without the initial SystemMessage).
    It keeps the first HumanMessage (original query) and the last `num_recent_to_keep` messages
    from this core history, injecting a condensation note.
    """
    if not full_history: # full_history here is actually core_history
        return []

    first_human_message: Optional[HumanMessage] = None
    for msg in full_history: # Iterate over the provided core_history
        if isinstance(msg, HumanMessage):
            first_human_message = msg
            break
    
    # If history is too short or no initial human query found in core_history,
    # return core_history as is. The calling function (assistant) will prepend SystemMessage.
    # Threshold considers: first_human (1) + condensation_note (1) + num_recent_to_keep
    if first_human_message is None or len(full_history) < (1 + 1 + num_recent_to_keep):
        return full_history

    # Pruning is needed for the core_history
    recent_messages_from_core = full_history[-num_recent_to_keep:]
    
    pruned_core_list: List[AnyMessage] = []
    
    # Add the first human message
    pruned_core_list.append(first_human_message)
    
    # Add condensation note
    pruned_core_list.append(
        AIMessage(content="[System note: To manage context length, earlier parts of the conversation have been omitted. The original query and the most recent interactions are preserved.]")
    )
    
    # Add recent messages, ensuring not to duplicate the first_human_message if it's in the recent slice
    for msg in recent_messages_from_core:
        if msg is not first_human_message: # Check object identity
            pruned_core_list.append(msg)
            
    return pruned_core_list

def assistant(state: AgentState) -> Dict[str, Any]:
    """Assistant node that processes messages and decides on next action."""
    from langchain_core.messages import AIMessage  # Add import at the start of the function
    
    print("Assistant Called...\n\n")
    
    full_current_history = state["messages"]
    iteration_count = state.get("iteration_count", 0)
    iteration_count += 1 # Increment for the current call
    print(f"Current Iteration: {iteration_count}")

    # Prepare messages for the LLM
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # Core history excludes any SystemMessages found in the accumulated history.
    # The pruning function operates on this core history.
    core_history = [msg for msg in full_current_history if not isinstance(msg, SystemMessage)]

    llm_input_core_messages: List[AnyMessage]

    # Prune if it's time (e.g., after every 5th completed iteration, so check for current iteration 6, 11, etc.)
    # Iteration 1-5: no pruning. Iteration 6: prune.
    if iteration_count > 5 and (iteration_count - 1) % 5 == 0:
        print(f"Pruning message history for LLM call at iteration {iteration_count}.")
        llm_input_core_messages = prune_messages_for_llm(core_history, num_recent_to_keep=6)
    else:
        llm_input_core_messages = core_history
    
    # Combine system message with the (potentially pruned) core messages
    messages_for_llm = [system_msg] + llm_input_core_messages
    
    # Get response from the assistant
    try:
        response = chat_with_tools.invoke(messages_for_llm, stop=["Observation:"])
        print(f"Assistant response type: {type(response)}")
        
        # Check for empty response
        if response is None or not hasattr(response, 'content') or not response.content or len(response.content.strip()) < 20:
            print("Warning: Received empty or very short response from LLM")
            
            # Create a new response object if it's None
            if response is None:
                response = AIMessage(content="")
                print("Created new AIMessage object for None response")
            
            # Check what kind of information we have in the last observation
            last_observation = None
            for msg in reversed(core_history):
                if isinstance(msg, AIMessage) and "Observation:" in msg.content:
                    last_observation = msg.content
                    break
                    
            # Create an appropriate fallback response
            if last_observation and "python_code" in state.get("current_tool", ""):
                # If last tool was Python code, try to formulate a reasonable next step
                print("Creating fallback response for empty response after Python code execution")
                fallback_content = (
                    "Thought: I've analyzed the results of the code execution. Based on the observations, "
                    "I need to determine the final answer.\n\n"
                    "Final Answer: "
                )
                
                # Look for common patterns in Python output that might indicate results
                import re
                result_match = re.search(r'(\[.+?\]|\{.+?\}|[A-Za-z,\s]+)$', last_observation)
                if result_match:
                    fallback_content += result_match.group(1).strip()
                else:
                    # Generic fallback based on the executed code
                    fallback_content += "Based on the code analysis, but the exact result was unclear from the output."
            else:
                # Generic fallback suggesting a reasonable next search
                print("Creating generic fallback response for empty LLM response")
                
                # Make sure we have a valid query
                query = ""
                if full_current_history and hasattr(full_current_history[0], 'content'):
                    query = full_current_history[0].content[:100]  # Limit to first 100 chars
                else:
                    query = "additional information for this question"
                    
                fallback_content = (
                    "Thought: I need more information to answer this question correctly. Let me search for additional details.\n\n"
                    "Action: \n```json\n"
                    "{\n"
                    '  "action": "tavily_search",\n'
                    f'  "action_input": {{"query": "{query}", "search_depth": "comprehensive"}}\n'
                    "}\n```"
                )
                
            # Use our fallback content instead of the empty response
            response.content = fallback_content
            print(f"Created fallback response: {fallback_content[:100]}...")
        else:
            content_preview = response.content[:300].replace('\n', ' ')
            print(f"Response content (first 300 chars): {content_preview}...")
    except Exception as e:
        print(f"Error in LLM invocation: {str(e)}")
        # Create a fallback response in case of LLM errors
        response = AIMessage(content=(
            "Thought: I encountered an error in processing. Let me try to proceed with what I know.\n\n"
            "Final Answer: Unable to complete the task due to an error in processing."
        ))
    
    # Extract the action JSON from the response text
    action_json = extract_json_from_text(response.content)
    print(f"Extracted action JSON: {action_json}")
    
    assistant_response_message = AIMessage(content=response.content)
    
    state_update: Dict[str, Any] = {
        "messages": [assistant_response_message], 
        "iteration_count": iteration_count
    }
    
    if action_json and "action" in action_json and "action_input" in action_json:
        tool_name = action_json["action"]
        tool_input = action_json["action_input"]
        
        # Handle nested JSON issue - check if any value in action_input is a JSON string
        if isinstance(tool_input, dict):
            for key, value in tool_input.items():
                if isinstance(value, str) and value.strip().startswith("{"):
                    try:
                        nested_json = json.loads(value)
                        if isinstance(nested_json, dict) and "action" in nested_json and "action_input" in nested_json:
                            # This is a nested structure, use the inner one
                            tool_name = nested_json["action"]
                            tool_input = nested_json["action_input"]
                            print(f"Unwrapped nested JSON. New tool: {tool_name}")
                            print(f"New tool input: {tool_input}")
                            break
                    except json.JSONDecodeError:
                        continue
        
        print(f"Using tool: {tool_name}")
        print(f"Tool input: {tool_input}")
        
        tool_call_id = f"call_{random.randint(1000000, 9999999)}"
        
        state_update["current_tool"] = tool_name
        state_update["action_input"] = tool_input
    else:
        print("No tool action found or 'Final Answer' detected in response.")
        state_update["current_tool"] = None
        state_update["action_input"] = None
        
    return state_update

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, handling markdown code blocks and other formats."""
    try:
        import re
        import json
        
        # Return empty if text is None or very short (less than 10 chars)
        if not text or len(text.strip()) < 10:
            print("Warning: Empty or very short text input to JSON extraction")
            return None
        
        print(f"Attempting to extract JSON from text: {text[:200]}...")
        
        # First, clean up the text to handle specific patterns that might confuse parsing
        text = text.replace('\\n', '\n').replace('\\"', '"')
        
        # Case 1: "Final Answer:" detection - if present, return None to indicate we should end
        if "Final Answer:" in text:
            print("Detected 'Final Answer' - no tool action needed")
            return None
        
        # Case 2: Extract direct python dictionary representation without JSON formatting
        if "action_input" in text and not '{"action"' in text and not '{"action_input"' in text:
            # Try regex to extract a Python dict-like structure
            action_match = re.search(r"action:\s*(\w+)", text, re.IGNORECASE)
            input_match = re.search(r"action_input:\s*(\{.+?\})", text, re.DOTALL | re.IGNORECASE)
            
            if action_match and input_match:
                action = action_match.group(1).strip()
                try:
                    action_input = eval(input_match.group(1))  # Be careful with eval!
                    if isinstance(action_input, dict):
                        return {"action": action, "action_input": action_input}
                except:
                    pass
        
        # Pattern 1: Look for "Action:" followed by a markdown code block
        action_match = re.search(r"Action:\s*```(?:python|json)?\s*(.*?)```", text, re.DOTALL)
        if action_match:
            action_content = action_match.group(1).strip()
            print(f"Found action content from markdown block: {action_content[:100]}...")
            
            # Try to parse as JSON first
            try:
                parsed_json = json.loads(action_content)
                if "action" in parsed_json and "action_input" in parsed_json:
                    return parsed_json
            except json.JSONDecodeError:
                # If it's Python code, create action structure
                if "=" in action_content or "import" in action_content or "print" in action_content:
                    print("Detected Python code, formatting as action_input")
                    return {
                        "action": "python_code",
                        "action_input": {"code": action_content}
                    }
        
        # Pattern 2: Look for regular markdown code blocks
        code_blocks = re.findall(r"```(?:json|python)?(.+?)```", text, re.DOTALL)
        for block in code_blocks:
            block = block.strip()
            print(f"Processing code block: {block[:100]}...")
            
            # Try to parse as JSON
            try:
                parsed = json.loads(block)
                if "action" in parsed and "action_input" in parsed:
                    print(f"Successfully parsed JSON block: {parsed}")
                    return parsed
            except json.JSONDecodeError:
                # If it's Python code, create action structure
                if "=" in block or "import" in block or "print" in block or "def " in block:
                    print("Detected Python code in block, formatting as action_input")
                    return {
                        "action": "python_code",
                        "action_input": {"code": block}
                    }
        
        # Pattern 3: Direct JSON object ({...}) in the text
        json_matches = re.findall(r"\{[\s\S]*?\}", text)
        for json_str in json_matches:
            try:
                parsed = json.loads(json_str)
                if "action" in parsed and "action_input" in parsed:
                    print(f"Found valid JSON object: {parsed}")
                    return parsed
            except json.JSONDecodeError:
                    continue
        
        # Pattern 4: Look for patterns like 'action': 'tool_name', 'action_input': {...}
        action_pattern = re.search(r"['\"](action)['\"]:\s*['\"](\w+)['\"]", text)
        action_input_pattern = re.search(r"['\"](action_input)['\"]:\s*(\{.+\})", text, re.DOTALL)
        
        if action_pattern and action_input_pattern:
            action = action_pattern.group(2)
            action_input_str = action_input_pattern.group(2)
            
            try:
                action_input = json.loads(action_input_str)
                return {
                    "action": action,
                    "action_input": action_input
                }
            except json.JSONDecodeError:
                pass
        
        # Pattern 5: Look for simple text patterns like "I need to use tool X to search for Y"
        tool_patterns = [
            (r"(?:use|using|need to use|should use|will use)(?:\s+the)?\s+(\w+)(?:\s+tool)?\s+to\s+(?:search|find|look up|research)(?:\s+for)?\s+['\"](.*?)['\"]", 
             lambda m: {"action": m.group(1).lower(), "action_input": {"query": m.group(2)}}),
            (r"(?:use|using|need to use|should use|will use)(?:\s+the)?\s+(\w+)(?:\s+tool)?\s+to\s+(?:search|find|look up|research)\s+(?:for\s+)?(.+?)(?=\.|$)", 
             lambda m: {"action": m.group(1).lower(), "action_input": {"query": m.group(2).strip()}}),
            (r"(?:use|using|need to use|should use|will use)(?:\s+the)?\s+(\w+)(?:\s+tool)?\s+on\s+['\"](.*?)['\"]", 
             lambda m: {"action": m.group(1).lower(), "action_input": {"query": m.group(2)}})
        ]
        
        for pattern, formatter in tool_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    result = formatter(match)
                    # Map common words to actual tool names
                    tool_mapping = {
                        "tavily": "tavily_search",
                        "wikipedia": "wikipedia_search",
                        "arxiv": "arxiv_search",
                        "web": "tavily_search",
                        "python": "python_code",
                        "excel": "excel_to_text",
                        "youtube": "process_youtube_video",
                        "webpage": "webpage_scrape",
                        "scrape": "webpage_scrape",
                        "pdf": "process_online_document",
                        "document": "process_online_document",
                        "online": "process_online_document"
                    }
                    
                    if result["action"].lower() in tool_mapping:
                        result["action"] = tool_mapping[result["action"].lower()]
                        
                    print(f"Extracted tool action using pattern: {result}")
                    return result
                except Exception as e:
                    print(f"Error formatting pattern match: {e}")
        
        # Fallback: If we detect thinking about a specific topic, suggest a related tool
        fallback_patterns = [
            (r"(?:I need|I should|I will|let me|I can)(?:\s+(?:to))?\s+(?:search|find|look for)\s+(?:information|details|data)?\s+(?:about|on|regarding)?\s+(.+?)(?=\.|$)",
             lambda m: {"action": "tavily_search", "action_input": {"query": m.group(1).strip()}}),
            (r"(?:I will|I should|let me)(?:\s+now)?\s+(?:search|check)(?:\s+on)?\s+wikipedia\s+(?:for)?\s+(.+?)(?=\.|$)",
             lambda m: {"action": "wikipedia_search", "action_input": {"query": m.group(1).strip()}}),
            (r"(?:I need|I should|I will|let me)(?:\s+to)?\s+(?:execute|run|write|use)\s+(?:some|a)?\s+python\s+(?:code)?",
             lambda m: {"action": "python_code", "action_input": {"code": "# Example code\nprint('Please specify Python code to execute')"}})
        ]
        
        for pattern, formatter in fallback_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    result = formatter(match)
                    print(f"Used fallback pattern to suggest tool: {result}")
                    return result
                except Exception as e:
                    print(f"Error in fallback pattern: {e}")
        
        print("Could not extract valid JSON from text using any pattern")
        
        # Last resort fallback: If we've tried everything and failed, and the response is long
        # enough to suggest the model is doing work but not formatting correctly, try to 
        # extract any query-like content and suggest a tavily search
        if len(text) > 200 and "search" in text.lower():
            # Extract a potential query - look for sentences with search-related terms
            search_sentences = re.findall(r'[^.!?]*(?:search|find|look for|investigate|research)[^.!?]*[.!?]', text)
            if search_sentences:
                best_sentence = max(search_sentences, key=len)
                # Clean up the query
                query = re.sub(r'(?:I will|I should|I need to|I want to|Let me|I\'ll|I can)\s+(?:search|find|look for|investigate|research)(?:\s+for)?\s+', '', best_sentence)
                query = query.strip('.!? \t\n')
                if query and len(query) > 5:
                    print(f"Last resort fallback: suggesting Tavily search for: {query}")
                    return {"action": "tavily_search", "action_input": {"query": query}}
        
        return None
        
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

# Comment out the web_search_node function
# def web_search_node(state: AgentState) -> Dict[str, Any]:
#     """Node that executes the web search tool."""
#     print("Web Search Tool Called...\n\n")
#     
#     # Extract tool arguments
#     action_input = state.get("action_input", {})
#     print(f"Web search action_input: {action_input}")
#     
#     # Try different ways to extract the query
#     query = ""
#     if isinstance(action_input, dict):
#         query = action_input.get("query", "")
#     elif isinstance(action_input, str):
#         query = action_input
#     
#     print(f"Searching for: '{query}'")
#     
#     # Call the search function with retry logic
#     result = safe_web_search(query)
#     print(f"Search result: {result}")  # Print the full result for debugging
#     
#     # Check if we hit rate limits and add a helpful note
#     if "rate limit" in result.lower() or "ratelimit" in result.lower():
#         result += "\n\nNote: You can use your internal knowledge to provide a response since the search is rate limited."
#     
#     # Format the observation to continue the ReAct cycle
#     # Don't include "Observation:" as the assistant is stopped at this token
#     observation = result
#     
#     # Create a tool message with the result
#     tool_message = AIMessage(
#         content=f"Observation: {observation}"
#     )
#     
#     # Print the observation that will be sent back to the assistant
#     print("\n=== TOOL OBSERVATION ===")
#     print(tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content)
#     print("=== END OBSERVATION ===\n")
#     
#     # Return the updated state
#     return {
#         "messages": state["messages"] + [tool_message],
#         "current_tool": None,  # Reset the current tool
#         "action_input": None   # Clear the action input
#     }

def python_code_node(state: AgentState) -> Dict[str, Any]:
    """Node that executes Python code."""
    print("Python Code Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Python code action_input: {action_input}")
    print(f"Action input type: {type(action_input)}")
    
    # Get the code string
    code = ""
    if isinstance(action_input, dict):
        code = action_input.get("code", "")
    elif isinstance(action_input, str):
        code = action_input
    
    print(f"Original code field (first 100 chars): {code[:100]}")
    
    def extract_code_from_json(json_str):
        """Recursively extract code from nested JSON structures."""
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                # Check for direct code field
                if "code" in parsed:
                    return parsed["code"]
                # Check for nested action_input structure
                if "action_input" in parsed:
                    inner_input = parsed["action_input"]
                    if isinstance(inner_input, dict):
                        if "code" in inner_input:
                            return inner_input["code"]
                        # If inner_input is also JSON string, recurse
                        if isinstance(inner_input.get("code", ""), str) and inner_input["code"].strip().startswith("{"):
                            return extract_code_from_json(inner_input["code"])
            return json_str
        except:
            return json_str
    
    # Handle nested JSON structures
    if isinstance(code, str) and code.strip().startswith("{"):
        code = extract_code_from_json(code)
        print("Extracted code from JSON structure")
    
    print(f"Final code to execute: {code[:100]}...")
    
    # Execute the code
    try:
        result = run_python_code(code)
        print(f"Code execution result: {result}")
        
        # Format the observation
        tool_message = AIMessage(
            content=f"Observation: {result.strip()}"
        )
        
        # Print the observation that will be sent back to the assistant
        print("\n=== TOOL OBSERVATION ===")
        content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
        print(content_preview)
        print("=== END OBSERVATION ===\n")
        
        # Return the updated state
        return {
            "messages": state["messages"] + [tool_message],
            "current_tool": None,  # Reset the current tool
            "action_input": None   # Clear the action input
        }
    except Exception as e:
        error_message = f"Error executing Python code: {str(e)}"
        print(error_message)
        tool_message = AIMessage(content=f"Observation: {error_message}")
        return {
            "messages": state["messages"] + [tool_message],
            "current_tool": None,
            "action_input": None
        }

def webpage_scrape_node(state: AgentState) -> Dict[str, Any]:
    """Node that scrapes content from a specific webpage URL."""
    print("Webpage Scrape Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Webpage scrape action_input: {action_input}")
    
    # Try different ways to extract the URL
    url = ""
    if isinstance(action_input, dict):
        url = action_input.get("url", "")
    elif isinstance(action_input, str):
        url = action_input
    
    print(f"Scraping URL: '{url}'")
    
    # Safety check - don't run with empty URL
    if not url:
        result = "Error: No URL provided. Please provide a valid URL to scrape."
    else:
        # Call the webpage scraping function
        result = scrape_webpage(url)
    
    print(f"Scraping result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    # Always prefix with "Observation:" for consistency in the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def wikipedia_search_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes Wikipedia search requests."""
    print("Wikipedia Search Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Wikipedia search action_input: {action_input}")
    
    # Extract query and num_results
    query = ""
    num_results = 3  # Default
    
    if isinstance(action_input, dict):
        query = action_input.get("query", "")
        if "num_results" in action_input:
            try:
                num_results = int(action_input["num_results"])
            except:
                print("Invalid num_results, using default")
    elif isinstance(action_input, str):
        query = action_input
    
    print(f"Searching Wikipedia for: '{query}' (max results: {num_results})")
    
    # Safety check - don't run with empty query
    if not query:
        result = "Error: No search query provided. Please provide a valid query for Wikipedia search."
    else:
        # Call the Wikipedia search function
        result = wikipedia_search(query, num_results)
    
    print(f"Wikipedia search result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def tavily_search_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes Tavily search requests."""
    print("Tavily Search Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Tavily search action_input: {action_input}")
    
    # Extract query and search_depth
    query = ""
    search_depth = "basic"  # Default
    
    if isinstance(action_input, dict):
        query = action_input.get("query", "")
        if "search_depth" in action_input:
            depth = action_input["search_depth"]
            if depth in ["basic", "comprehensive"]:
                search_depth = depth
    elif isinstance(action_input, str):
        query = action_input
    
    print(f"Searching Tavily for: '{query}' (depth: {search_depth})")
    
    # Safety check - don't run with empty query
    if not query:
        result = "Error: No search query provided. Please provide a valid query for Tavily search."
    else:
        # Call the Tavily search function
        result = tavily_search(query, search_depth)
    
    print(f"Tavily search result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def arxiv_search_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes ArXiv search requests."""
    print("ArXiv Search Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"ArXiv search action_input: {action_input}")
    
    # Extract query and max_results
    query = ""
    max_results = 5  # Default
    
    if isinstance(action_input, dict):
        query = action_input.get("query", "")
        if "max_results" in action_input:
            try:
                max_results = int(action_input["max_results"])
                if max_results <= 0 or max_results > 10:
                    max_results = 5  # Reset to default if out of range
            except:
                print("Invalid max_results, using default")
    elif isinstance(action_input, str):
        query = action_input
    
    print(f"Searching ArXiv for: '{query}' (max results: {max_results})")
    
    # Safety check - don't run with empty query
    if not query:
        result = "Error: No search query provided. Please provide a valid query for ArXiv search."
    else:
        # Call the ArXiv search function
        result = arxiv_search(query, max_results)
    
    print(f"ArXiv search result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def supabase_operation_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes Supabase database operations."""
    print("Supabase Operation Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Supabase operation action_input: {action_input}")
    
    # Extract required parameters
    operation_type = ""
    table = ""
    data = None
    filters = None
    
    if isinstance(action_input, dict):
        operation_type = action_input.get("operation_type", "")
        table = action_input.get("table", "")
        data = action_input.get("data")
        filters = action_input.get("filters")
    
    print(f"Supabase operation: {operation_type} on table {table}")
    
    # Safety check
    if not operation_type or not table:
        result = "Error: Both operation_type and table are required. operation_type should be one of: insert, select, update, delete"
    else:
        # Call the Supabase operation function
        result = supabase_operation(operation_type, table, data, filters)
    
    print(f"Supabase operation result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def excel_to_text_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes Excel to Markdown table conversions."""
    print("Excel to Text Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Excel to text action_input: {action_input}")
    
    # Extract required parameters
    excel_path = ""
    sheet_name = None
    file_content = None
    
    if isinstance(action_input, dict):
        excel_path = action_input.get("excel_path", "")
        sheet_name = action_input.get("sheet_name")
        
        # Check if there's attached file content (base64 encoded) directly in the action_input
        if "file_content" in action_input and action_input["file_content"]:
            try:
                file_content = base64.b64decode(action_input["file_content"])
                print(f"Decoded attached file content, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error decoding file content from action_input: {e}")
        
        # Check if we should use a file from the attachments dictionary
        if not file_content and excel_path and "attachments" in state and excel_path in state["attachments"]:
            try:
                attachment_data = state["attachments"][excel_path]
                if attachment_data:  # Make sure it's not empty
                    file_content = base64.b64decode(attachment_data)
                    print(f"Using attachment '{excel_path}' from state, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error using attachment {excel_path}: {e}")
    
    print(f"Excel to text: path={excel_path}, sheet={sheet_name or 'default'}, has_attachment={file_content is not None}")
    
    # Safety check
    if not excel_path and not file_content:
        result = "Error: Either Excel file path or file content is required"
    elif not file_content:
        # If we have a path but no content, check if it's a local file that exists
        local_file_path = Path(excel_path).expanduser().resolve()
        if local_file_path.is_file():
            # Local file exists, use it directly
            result = excel_to_text(str(local_file_path), sheet_name, None)
        else:
            # No file content and path doesn't exist as a local file
            result = f"Error: Excel file not found at {local_file_path} and no attachment data available"
    else:
        # We have file content, use it
        result = excel_to_text(excel_path, sheet_name, file_content)
    
    print(f"Excel to text result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

# Add a new node function for processing YouTube videos
def process_youtube_video_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes YouTube videos."""
    print("YouTube Video Processing Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"YouTube video processing action_input: {action_input}")
    
    # Extract URL and other parameters
    url = ""
    summarize = True  # Default
    
    if isinstance(action_input, dict):
        url = action_input.get("url", "")
        # Check if summarize parameter exists and is a boolean
        if "summarize" in action_input:
            try:
                summarize = bool(action_input["summarize"])
            except:
                print("Invalid summarize parameter, using default (True)")
    elif isinstance(action_input, str):
        # If action_input is just a string, assume it's the URL
        url = action_input
    
    print(f"Processing YouTube video: '{url}' (summarize: {summarize})")
    
    # Safety check - don't run with empty URL
    if not url:
        result = "Error: No URL provided. Please provide a valid YouTube URL."
    elif not url.startswith(("http://", "https://")) or not ("youtube.com" in url or "youtu.be" in url):
        result = f"Error: Invalid YouTube URL format: {url}. Please provide a valid URL starting with http:// or https:// and containing youtube.com or youtu.be."
    else:
        # Call the YouTube processing function
        try:
            result = process_youtube_video(url, summarize)
        except Exception as e:
            result = f"Error processing YouTube video: {str(e)}\n\nThis could be due to:\n- The video is private or has been removed\n- Network connectivity issues\n- YouTube API changes\n- Rate limiting"
    
    print(f"YouTube processing result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

# Add after the existing tool nodes:
def transcribe_audio_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes audio transcription requests."""
    print("Audio Transcription Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Audio transcription action_input: {action_input}")
    
    # Extract required parameters
    audio_path = ""
    language = None
    file_content = None
    
    if isinstance(action_input, dict):
        audio_path = action_input.get("audio_path", "")
        language = action_input.get("language")
        
        # Check if there's attached file content (base64 encoded) directly in the action_input
        if "file_content" in action_input and action_input["file_content"]:
            try:
                file_content = base64.b64decode(action_input["file_content"])
                print(f"Decoded attached audio file content, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error decoding file content from action_input: {e}")
        
        # Check if we should use a file from the attachments dictionary
        if not file_content and audio_path and "attachments" in state and audio_path in state["attachments"]:
            try:
                attachment_data = state["attachments"][audio_path]
                if attachment_data:  # Make sure it's not empty
                    file_content = base64.b64decode(attachment_data)
                    print(f"Using attachment '{audio_path}' from state, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error using attachment {audio_path}: {e}")
    
    print(f"Audio transcription: path={audio_path}, language={language or 'auto-detect'}, has_attachment={file_content is not None}")
    
    # Safety check
    if not audio_path:
        result = "Error: Audio file path is required"
    elif not file_content:
        # If we have a path but no content, check if it's a local file that exists
        local_file_path = Path(audio_path).expanduser().resolve()
        if local_file_path.is_file():
            # Local file exists, use it directly
            result = transcribe_audio(str(local_file_path), None, language)
        else:
            # No file content and path doesn't exist as a local file
            result = f"Error: Audio file not found at {local_file_path} and no attachment data available"
    else:
        # We have file content, use it
        result = transcribe_audio(audio_path, file_content, language)
    
    print(f"Audio transcription result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def process_image_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes image analysis requests."""
    print("Image Processing Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Image processing action_input: {action_input}")
    
    # Extract required parameters
    image_path = ""
    image_url = None
    analyze_content = True  # Default to true
    file_content = None
    
    if isinstance(action_input, dict):
        image_path = action_input.get("image_path", "")
        image_url = action_input.get("image_url")
        
        # Check if analyze_content parameter exists and is a boolean
        if "analyze_content" in action_input:
            try:
                analyze_content = bool(action_input["analyze_content"])
            except:
                print("Invalid analyze_content parameter, using default (True)")
        
        # Check if there's attached file content (base64 encoded) directly in the action_input
        if "file_content" in action_input and action_input["file_content"]:
            try:
                file_content = base64.b64decode(action_input["file_content"])
                print(f"Decoded attached image file content, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error decoding file content from action_input: {e}")
        
        # Check if we should use a file from the attachments dictionary
        if not file_content and image_path and "attachments" in state and image_path in state["attachments"]:
            try:
                attachment_data = state["attachments"][image_path]
                if attachment_data:  # Make sure it's not empty
                    file_content = base64.b64decode(attachment_data)
                    print(f"Using attachment '{image_path}' from state, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error using attachment {image_path}: {e}")
    
    print(f"Image processing: path={image_path}, url={image_url or 'None'}, analyze_content={analyze_content}, has_attachment={file_content is not None}")
    
    # Safety check
    if not image_path and not image_url and not file_content:
        result = "Error: Either image path, image URL, or file content is required"
    elif not file_content and not image_url:
        # If we have a path but no content, check if it's a local file that exists
        local_file_path = Path(image_path).expanduser().resolve()
        if local_file_path.is_file():
            # Local file exists, use it directly
            result = process_image(str(local_file_path), image_url, None, analyze_content)
        else:
            # No file content and path doesn't exist as a local file
            result = f"Error: Image file not found at {local_file_path} and no attachment data available"
    else:
        # We have file content or URL, use it
        result = process_image(image_path, image_url, file_content, analyze_content)
    
    print(f"Image processing result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def read_file_node(state: AgentState) -> Dict[str, Any]:
    """Node that reads text file contents."""
    print("File Reading Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"File reading action_input: {action_input}")
    
    # Extract required parameters
    file_path = ""
    line_start = None
    line_end = None
    file_content = None
    
    if isinstance(action_input, dict):
        file_path = action_input.get("file_path", "")
        
        # Check if line range parameters exist
        if "line_start" in action_input:
            try:
                line_start = int(action_input["line_start"])
            except:
                print("Invalid line_start parameter, using default (None)")
        
        if "line_end" in action_input:
            try:
                line_end = int(action_input["line_end"])
            except:
                print("Invalid line_end parameter, using default (None)")
        
        # Check if there's attached file content (base64 encoded) directly in the action_input
        if "file_content" in action_input and action_input["file_content"]:
            try:
                file_content = base64.b64decode(action_input["file_content"])
                print(f"Decoded attached file content, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error decoding file content from action_input: {e}")
        
        # Check if we should use a file from the attachments dictionary
        if not file_content and file_path and "attachments" in state and file_path in state["attachments"]:
            try:
                attachment_data = state["attachments"][file_path]
                if attachment_data:  # Make sure it's not empty
                    file_content = base64.b64decode(attachment_data)
                    print(f"Using attachment '{file_path}' from state, size: {len(file_content)} bytes")
            except Exception as e:
                print(f"Error using attachment {file_path}: {e}")
    
    print(f"File reading: path={file_path}, line_range={line_start}-{line_end if line_end else 'end'}, has_attachment={file_content is not None}")
    
    # Safety check
    if not file_path:
        result = "Error: File path is required"
    elif not file_content:
        # If we have a path but no content, check if it's a local file that exists
        local_file_path = Path(file_path).expanduser().resolve()
        if local_file_path.is_file():
            # Local file exists, use it directly
            result = read_file(str(local_file_path), None, line_start, line_end)
        else:
            # No file content and path doesn't exist as a local file
            result = f"Error: File not found at {local_file_path} and no attachment data available"
    else:
        # We have file content, use it
        result = read_file(file_path, file_content, line_start, line_end)
    
    print(f"File reading result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
    print("=== END OBSERVATION ===\n")
    
    # Return the updated state
    return {
        "messages": state["messages"] + [tool_message],
        "current_tool": None,  # Reset the current tool
        "action_input": None   # Clear the action input
    }

def process_online_document_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes online PDFs and images."""
    print("Online Document Processing Tool Called...\n\n")
    
    # Extract tool arguments
    action_input = state.get("action_input", {})
    print(f"Online document processing action_input: {action_input}")
    
    # Extract URL and document type
    url = ""
    doc_type = "auto"  # Default to auto-detection
    
    if isinstance(action_input, dict):
        url = action_input.get("url", "")
        doc_type = action_input.get("doc_type", "auto")
    elif isinstance(action_input, str):
        url = action_input
    
    print(f"Processing online document: '{url}' (type: {doc_type})")
    
    # Safety check - don't run with empty URL
    if not url:
        result = "Error: No URL provided. Please provide a valid URL to process."
    elif not url.startswith(("http://", "https://")):
        result = f"Error: Invalid URL format: {url}. Please provide a valid URL starting with http:// or https://."
    else:
        # Call the online document processing function
        try:
            result = process_online_document(url, doc_type)
        except Exception as e:
            result = f"Error processing online document: {str(e)}\n\nThis could be due to:\n- The document is not accessible\n- Network connectivity issues\n- Unsupported document type\n- Rate limiting"
    
    print(f"Online document processing result length: {len(result)}")
    
    # Format the observation to continue the ReAct cycle
    tool_message = AIMessage(
        content=f"Observation: {result.strip()}"
    )
    
    # Print the observation that will be sent back to the assistant
    print("\n=== TOOL OBSERVATION ===")
    content_preview = tool_message.content[:500] + "..." if len(tool_message.content) > 500 else tool_message.content
    print(content_preview)
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
    
    if tool == "python_code":
        return "python_code"
    elif tool == "webpage_scrape":
        return "webpage_scrape"
    elif tool == "wikipedia_search":
        return "wikipedia_search"
    elif tool == "tavily_search":
        return "tavily_search"
    elif tool == "arxiv_search":
        return "arxiv_search"
    elif tool == "supabase_operation":
        return "supabase_operation"
    elif tool == "excel_to_text":
        return "excel_to_text"
    elif tool == "process_youtube_video":
        return "process_youtube_video"
    elif tool == "transcribe_audio":
        return "transcribe_audio"
    elif tool == "process_image":
        return "process_image"
    elif tool == "read_file":
        return "read_file"
    elif tool == "process_online_document":
        return "process_online_document"
    else:
        return "end"

# Create the graph
def create_agent_graph() -> StateGraph:
    """Create the complete agent graph with individual tool nodes."""
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("python_code", python_code_node)
    builder.add_node("webpage_scrape", webpage_scrape_node)
    builder.add_node("wikipedia_search", wikipedia_search_node)
    builder.add_node("tavily_search", tavily_search_node)
    builder.add_node("arxiv_search", arxiv_search_node)
    builder.add_node("supabase_operation", supabase_operation_node)
    builder.add_node("excel_to_text", excel_to_text_node)
    builder.add_node("process_youtube_video", process_youtube_video_node)
    builder.add_node("transcribe_audio", transcribe_audio_node)
    builder.add_node("process_image", process_image_node)
    builder.add_node("read_file", read_file_node)
    builder.add_node("process_online_document", process_online_document_node)

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
            "python_code": "python_code",
            "webpage_scrape": "webpage_scrape",
            "wikipedia_search": "wikipedia_search",
            "tavily_search": "tavily_search",
            "arxiv_search": "arxiv_search",
            "supabase_operation": "supabase_operation",
            "excel_to_text": "excel_to_text",
            "process_youtube_video": "process_youtube_video",
            "transcribe_audio": "transcribe_audio",
            "process_image": "process_image",
            "read_file": "read_file",
            "process_online_document": "process_online_document",
            "end": END
        }
    )
    
    # Tools always go back to assistant
    builder.add_edge("python_code", "assistant")
    builder.add_edge("webpage_scrape", "assistant")
    builder.add_edge("wikipedia_search", "assistant")
    builder.add_edge("tavily_search", "assistant")
    builder.add_edge("arxiv_search", "assistant")
    builder.add_edge("supabase_operation", "assistant")
    builder.add_edge("excel_to_text", "assistant")
    builder.add_edge("process_youtube_video", "assistant")
    builder.add_edge("transcribe_audio", "assistant")
    builder.add_edge("process_image", "assistant")
    builder.add_edge("read_file", "assistant")
    builder.add_edge("process_online_document", "assistant")
    
    # Compile the graph
    return builder.compile()

# Main agent class that integrates with your existing app.py
class TurboNerd:
    def __init__(self, max_iterations=25, apify_api_token=None):
        self.graph = create_agent_graph()
        self.tools = tools_config
        self.max_iterations = max_iterations  # Maximum iterations for the graph
        
        # Set Apify API token if provided
        if apify_api_token:
            os.environ["APIFY_API_TOKEN"] = apify_api_token
            print("Apify API token set successfully")
    
    def __call__(self, question: str, attachments: dict = None) -> str:
        """
        Process a question and return an answer.
        
        Args:
            question: The user's question text
            attachments: Optional dictionary of attachments with keys as names and values as base64-encoded content
        """
        # Process attachments if provided
        attachment_info = ""
        if attachments and isinstance(attachments, dict) and len(attachments) > 0:
            attachment_names = list(attachments.keys())
            attachment_info = f"\n\nI've attached the following files: {', '.join(attachment_names)}. "
            
            # Add different instructions based on detected file types
            excel_files = [name for name in attachment_names if name.endswith(('.xlsx', '.xls'))]
            if excel_files:
                attachment_info += f"Use the excel_to_text tool with the file_content parameter to process the Excel files."
        
        # Initialize the state with the question and attachment info
        question_with_attachments = question + attachment_info if attachment_info else question
        
        initial_state = {
            "messages": [HumanMessage(content=f"Question: {question_with_attachments}")],
            "current_tool": None,
            "action_input": None,
            "iteration_count": 0,  # Initialize iteration_count
            "attachments": attachments or {}  # Store attachments in the state
        }
        
        # Run the graph
        print(f"Starting graph execution with question: {question}")
        if attachments:
            print(f"Included attachments: {list(attachments.keys())}")
        
        try:
            # Set a reasonable recursion limit based on max_iterations
            result = self.graph.invoke(initial_state, {"recursion_limit": self.max_iterations})
            
            # Print the final state for debugging
            print(f"Final state keys: {result.keys()}")
            print(f"Final message count: {len(result['messages'])}")
            
            # Extract the final message and return just the final answer
            final_message = result["messages"][-1].content
            print("Final message: ", final_message)
            
            # Extract just the final answer part
            if "Final Answer:" in final_message:
                final_answer = final_message.split("Final Answer:", 1)[1].strip()
                return final_answer
            
            return final_message
        
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            # Otherwise return the error
            return f"I encountered an error while processing your question: {str(e)}"

# Example usage:
if __name__ == "__main__":
    agent = TurboNerd(max_iterations=25)
    response = agent("""The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places. TEMPP\excel.xlsx """)
    print("\nFinal Response:")
    print(response)

