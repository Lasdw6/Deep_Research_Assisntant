import os
from dotenv import load_dotenv
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
from urllib.parse import quote, urlparse
import sys
from bs4 import BeautifulSoup
import html2text

from apify_client import ApifyClient
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults  # For Tavily search
from supabase import create_client, Client

load_dotenv()

def run_python_code(code: str):
    """Execute Python code safely using exec() instead of subprocess."""
    # Check for potentially dangerous operations
    dangerous_operations = [
        "os.system", "os.popen", "os.unlink", "os.remove",
        "subprocess.run", "subprocess.call", "subprocess.Popen",
        "shutil.rmtree", "shutil.move", "shutil.copy",
        "open(", "file(", "eval(", "exec(", 
        "__import__", "input(", "raw_input(",
        "__builtins__", "globals(", "locals(",
        "compile(", "execfile(", "reload("
    ]
    
    # Safe imports that should be allowed
    safe_imports = {
        "import datetime", "import math", "import random", 
        "import statistics", "import collections", "import itertools",
        "import re", "import json", "import csv", "import numpy",
        "import pandas", "from math import", "from datetime import",
        "from statistics import", "from collections import",
        "from itertools import"
    }
    
    # Check for dangerous operations
    for dangerous_op in dangerous_operations:
        if dangerous_op in code:
            return f"Error: Code contains potentially unsafe operations: {dangerous_op}"
    
    # Check each line for imports
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            # Check if it's in our safe list
            is_safe = any(line.startswith(safe_import) for safe_import in safe_imports)
            # Also allow basic numpy/pandas imports
            is_safe = is_safe or line.startswith("import numpy") or line.startswith("import pandas")
            if not is_safe:
                return f"Error: Code contains potentially unsafe import: {line}"
    
    try:
        # Capture stdout to get print output
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Create a restricted globals environment
        restricted_globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                'chr': chr, 'dict': dict, 'dir': dir, 'divmod': divmod,
                'enumerate': enumerate, 'filter': filter, 'float': float,
                'format': format, 'hex': hex, 'int': int, 'len': len,
                'list': list, 'map': map, 'max': max, 'min': min, 'oct': oct,
                'ord': ord, 'pow': pow, 'print': print, 'range': range,
                'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
                'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                'type': type, 'zip': zip,
            }
        }
        
        # Allow safe modules
        import math
        import datetime
        import random
        import statistics
        import collections
        import itertools
        import re
        import json
        import csv
        
        restricted_globals['math'] = math
        restricted_globals['datetime'] = datetime
        restricted_globals['random'] = random
        restricted_globals['statistics'] = statistics
        restricted_globals['collections'] = collections
        restricted_globals['itertools'] = itertools
        restricted_globals['re'] = re
        restricted_globals['json'] = json
        restricted_globals['csv'] = csv
        
        # Try to import numpy and pandas if available
        try:
            import numpy as np
            restricted_globals['numpy'] = np
            restricted_globals['np'] = np
        except ImportError:
            pass
            
        try:
            import pandas as pd
            restricted_globals['pandas'] = pd
            restricted_globals['pd'] = pd
        except ImportError:
            pass
        
        # Create local scope
        local_scope = {}
        
        # Capture stdout
        captured_output = io.StringIO()
        
        # Execute the code with timeout simulation (not perfect but better than nothing)
        with redirect_stdout(captured_output):
            # Split code into lines and execute
            lines = code.strip().split('\n')
            last_line = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Check if this is the last meaningful line
                is_last = i == len(lines) - 1
                
                # Execute the line
                if is_last and not any(keyword in line for keyword in ['print', 'for', 'while', 'if', 'def', 'class', 'try', 'with']):
                    # If it's the last line and looks like an expression, store it
                    try:
                        # Try to evaluate as expression first
                        result = eval(line, restricted_globals, local_scope)
                        local_scope['_last_result'] = result
                        print(f"Result: {result}")
                    except:
                        # If that fails, execute as statement
                        exec(line, restricted_globals, local_scope)
                else:
                    exec(line, restricted_globals, local_scope)
        
        # Get the captured output
        output = captured_output.getvalue()
        
        if output.strip():
            return output.strip()
        else:
            # If no output but we have a last result, show it
            if '_last_result' in local_scope:
                return f"Result: {local_scope['_last_result']}"
            else:
                return "Code executed successfully with no output."
                
    except SyntaxError as e:
        return f"Syntax Error: {str(e)}"
    except NameError as e:
        return f"Name Error: {str(e)}"
    except ZeroDivisionError as e:
        return f"Zero Division Error: {str(e)}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

# Apify-based search function
# def apify_google_search(query: str, limit: int = 10) -> str:
#     """
#     Use Apify's Google Search Results Scraper to get search results
#     
#     Args:
#         query: The search query string
#         limit: Number of results to return (10, 20, 30, 40, 50, 100)
#         
#     Returns:
#         Formatted search results as a string
#     """
#     # You would need to provide a valid Apify API token
#     # You can get one by signing up at https://apify.com/
#     # Replace this with your actual Apify API token or set as environment variable
#     APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
#     
#     if not APIFY_API_TOKEN:
#         print("No Apify API token found. Using fallback search method.")
#         return fallback_search(query)
#     
#     try:
#         # Initialize the ApifyClient with API token
#         client = ApifyClient(APIFY_API_TOKEN)
#         
#         # Prepare the Actor input - convert limit to string as required by the API
#         run_input = {
#             "keyword": query,
#             "limit": str(limit),  # Convert to string as required by the API
#             "country": "US"
#         }
#         
#         # The Actor ID for the Google Search Results Scraper
#         ACTOR_ID = "563JCPLOqM1kMmbbP"
#         
#         print(f"Starting Apify search for: '{query}'")
#         
#         # Run the Actor and wait for it to finish (with timeout)
#         run = client.actor(ACTOR_ID).call(run_input=run_input, timeout_secs=60)
#         
#         if not run or not run.get("defaultDatasetId"):
#             print("Failed to get results from Apify actor")
#             return fallback_search(query)
#             
#         # Fetch Actor results from the run's dataset
#         results = []
#         for item in client.dataset(run["defaultDatasetId"]).iterate_items():
#             results.append(item)
#         
#         # Format and return the results
#         return format_search_results(results, query)
#         
#     except Exception as e:
#         print(f"Error using Apify: {str(e)}")
#         return fallback_search(query)

def scrape_webpage(url: str) -> str:
    """
    Safely scrape content from a specified URL.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Formatted webpage content as text
    """
    # Check if the URL is valid
    try:
        # Parse the URL to validate it
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return f"Error: Invalid URL format: {url}. Please provide a valid URL with http:// or https:// prefix."
        
        # Block potentially dangerous URLs
        blocked_domains = [
            "localhost", "127.0.0.1", "0.0.0.0", 
            "192.168.", "10.0.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
            "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", 
            "172.28.", "172.29.", "172.30.", "172.31."
        ]
        
        if any(domain in parsed_url.netloc for domain in blocked_domains):
            return f"Error: Access to internal/local URLs is blocked for security: {url}"
        
        print(f"Scraping URL: {url}")
        
        # Set user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        # Set a reasonable timeout to avoid hanging
        timeout = 10
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=timeout)
        
        # Check if request was successful
        if response.status_code != 200:
            return f"Error: Failed to fetch the webpage. Status code: {response.status_code}"
        
        # Use BeautifulSoup to parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements that are not relevant to content
        for script_or_style in soup(["script", "style", "iframe", "footer", "nav"]):
            script_or_style.decompose()
        
        # Get the page title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract the main content
        # First try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content')
        
        # If no main content area is found, use the entire body
        if not main_content:
            main_content = soup.body
        
        # Convert to plain text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_tables = False
        h.unicode_snob = True
        
        if main_content:
            text_content = h.handle(str(main_content))
        else:
            text_content = h.handle(response.text)
        
        # Limit content length to avoid overwhelming the model
        max_content_length = 99999999999
        if len(text_content) > max_content_length:
            text_content = text_content[:max_content_length] + "\n\n[Content truncated due to length...]"
        
        # Format the response
        result = f"Title: {title}\nURL: {url}\n\n{text_content}"
        
        return result
        
    except requests.exceptions.Timeout:
        return f"Error: Request timed out while trying to access {url}"
    except requests.exceptions.ConnectionError:
        return f"Error: Failed to connect to {url}. The site might be down or the URL might be incorrect."
    except requests.exceptions.RequestException as e:
        return f"Error requesting {url}: {str(e)}"
    except Exception as e:
        return f"Error scraping webpage {url}: {str(e)}"

# Comment out the format_search_results function (around line 180)
# def format_search_results(results: List[Dict], query: str) -> str:
#     """Format the search results into a readable string"""
#     if not results or len(results) == 0:
#         return f"No results found for query: {query}"
#     
#     print(f"Raw search results: {str(results)[:1000]}...")
#     
#     # Extract search results from the Apify output
#     formatted_results = f"Search results for '{query}':\n\n"
#     
#     # Check if results is a list of dictionaries or a dictionary with nested results
#     if isinstance(results, dict) and "results" in results:
#         items = results["results"]
#     elif isinstance(results, list):
#         items = results
#     else:
#         return f"Unable to process results for query: {query}"
#     
#     # Handle different Apify result formats
#     if len(items) > 0:
#         # Check the structure of the first item to determine format
#         first_item = items[0]
#         
#         # If item has 'organicResults', this is the format from some Apify actors
#         if isinstance(first_item, dict) and "organicResults" in first_item:
#             organic_results = first_item.get("organicResults", [])
#             for i, result in enumerate(organic_results[:10], 1):
#                 if "title" in result and "url" in result:
#                     formatted_results += f"{i}. {result['title']}\n"
#                     formatted_results += f"   URL: {result['url']}\n"
#                     if "snippet" in result:
#                         formatted_results += f"   {result['snippet']}\n"
#                     formatted_results += "\n"
#         else:
#             # Standard format with title/url/description
#             for i, result in enumerate(items[:10], 1):
#                 if "title" in result and "url" in result:
#                     formatted_results += f"{i}. {result['title']}\n"
#                     formatted_results += f"   URL: {result['url']}\n"
#                     if "description" in result:
#                         formatted_results += f"   {result['description']}\n"
#                     elif "snippet" in result:
#                         formatted_results += f"   {result['snippet']}\n"
#                     formatted_results += "\n"
#     
#     return formatted_results

# Comment out the fallback_search function (around line 220)
# def fallback_search(query: str) -> str:
#     """Fallback search method using DuckDuckGo when Apify is not available"""
#     try:
#         search_tool = DuckDuckGoSearchRun()
#         result = search_tool.invoke(query)
#         return "Observation: " + result
#     except Exception as e:
#         return f"Search error: {str(e)}. Please try a different query or method."

# Comment out the safe_web_search function (around line 230)
# def safe_web_search(query: str) -> str:
#     """Search the web safely with error handling and retry logic."""
#     if not query:
#         return "Error: No search query provided. Please specify what you want to search for."
#     
#     # Try using Apify first, if it fails it will use the fallback
#     return "Observation: " + apify_google_search(query)
#     
#     # The code below is kept for reference but won't be executed
#     max_retries = 3
#     backoff_factor = 1.5
#     
#     for attempt in range(max_retries):
#         try:
#             # Use the DuckDuckGoSearchRun tool
#             search_tool = DuckDuckGoSearchRun()
#             result = search_tool.invoke(query)
#             
#             # If we get an empty result, provide a helpful message
#             if not result or len(result.strip()) < 10:
#                 return f"The search for '{query}' did not return any useful results. Please try a more specific query or a different search engine."
#             
#             return "Observation: " + result
#             
#         except Exception as e:
#             # If we're being rate limited
#             if "Ratelimit" in str(e) or "429" in str(e):
#                 if attempt < max_retries - 1:
#                     wait_time = backoff_factor ** attempt
#                     print(f"Rate limited, waiting {wait_time:.2f} seconds before retrying...")
#                     time.sleep(wait_time)
#                 else:
#                     # On last attempt, return a helpful error
#                     error_msg = f"I'm currently unable to search for '{query}' due to service rate limits. "
#                     return error_msg
#             else:
#                 # For other types of errors
#                 return f"Error while searching for '{query}': {str(e)}"
#             
#     return f"Failed to search for '{query}' after multiple attempts due to rate limiting."

def wikipedia_search(query: str, num_results: int = 3) -> str:
    """
    Search Wikipedia for information about a specific query.
    
    Args:
        query: Search query
        num_results: Number of search results to return (default: 3)
        
    Returns:
        Formatted Wikipedia search results
    """
    try:
        # Validate input
        if not query or not isinstance(query, str):
            return "Error: Please provide a valid search query."
        
        # Ensure num_results is valid
        try:
            num_results = int(num_results)
            if num_results <= 0:
                num_results = 3  # Default to 3 if invalid
        except:
            num_results = 3  # Default to 3 if conversion fails
            
        print(f"Searching Wikipedia for: {query}")
        
        # Use WikipediaLoader from LangChain
        loader = WikipediaLoader(query=query, load_max_docs=num_results)
        docs = loader.load()
        
        if not docs:
            return f"No Wikipedia results found for '{query}'. Try refining your search."
        
        # Format the results
        formatted_results = f"Wikipedia search results for '{query}':\n\n"
        
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get('title', 'Unknown Title')
            source = doc.metadata.get('source', 'No URL')
            content = doc.page_content
            
            # Truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
                
            formatted_results += f"{i}. {title}\n"
            formatted_results += f"   URL: {source}\n"
            formatted_results += f"   {content}\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

def tavily_search(query: str, search_depth: str = "basic") -> str:
    """
    Search the web using the Tavily Search API.
    
    Args:
        query: Search query
        search_depth: Depth of search ('basic' or 'comprehensive')
        
    Returns:
        Formatted search results from Tavily
    """
    try:
        # Check for API key
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return "Error: Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
            
        # Validate input
        if not query or not isinstance(query, str):
            return "Error: Please provide a valid search query."
            
        # Validate search_depth
        if search_depth not in ["basic", "comprehensive"]:
            search_depth = "basic"  # Default to basic if invalid
            
        print(f"Searching Tavily for: {query} (depth: {search_depth})")
        
        # Initialize the Tavily search tool
        search = TavilySearchResults(api_key=tavily_api_key)
        
        # Execute the search
        results = search.invoke({"query": query, "search_depth": search_depth})
        
        if not results:
            return f"No Tavily search results found for '{query}'. Try refining your search."
            
        # Format the results
        formatted_results = f"Tavily search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result.get('title', 'No title')}\n"
            formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
            formatted_results += f"   {result.get('content', 'No content')}\n\n"
            
        return formatted_results
        
    except Exception as e:
        return f"Error searching with Tavily: {str(e)}"

def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Search ArXiv for scientific papers matching the query.
    
    Args:
        query: Search query for ArXiv
        max_results: Maximum number of results to return
        
    Returns:
        Formatted ArXiv search results
    """
    try:
        # Validate input
        if not query or not isinstance(query, str):
            return "Error: Please provide a valid search query."
            
        # Ensure max_results is valid
        try:
            max_results = int(max_results)
            if max_results <= 0 or max_results > 10:
                max_results = 5  # Default to 5 if invalid or too large
        except:
            max_results = 5  # Default to 5 if conversion fails
            
        print(f"Searching ArXiv for: {query}")
        
        # Use ArxivLoader from LangChain
        loader = ArxivLoader(
            query=query,
            load_max_docs=max_results,
            load_all_available_meta=True
        )
        
        docs = loader.load()
        
        if not docs:
            return f"No ArXiv papers found for '{query}'. Try refining your search."
            
        # Format the results
        formatted_results = f"ArXiv papers for '{query}':\n\n"
        
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            title = meta.get('Title', 'Unknown Title')
            url = meta.get('Entry ID', 'No URL')
            authors = meta.get('Authors', 'Unknown Authors')
            published = meta.get('Published', 'Unknown Date')
            
            formatted_results += f"{i}. {title}\n"
            formatted_results += f"   URL: {url}\n"
            formatted_results += f"   Authors: {authors}\n"
            formatted_results += f"   Published: {published}\n"
            
            # Add abstract, truncated if too long
            abstract = doc.page_content.replace('\n', ' ')
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            formatted_results += f"   Abstract: {abstract}\n\n"
            
        return formatted_results
        
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

def supabase_operation(operation_type: str, table: str, data: dict = None, filters: dict = None) -> str:
    """
    Perform operations on Supabase database.
    
    Args:
        operation_type: Type of operation ('insert', 'select', 'update', 'delete')
        table: Name of the table to operate on
        data: Data to insert/update (for insert/update operations)
        filters: Filters for select/update/delete operations (e.g., {"id": 1})
        
    Returns:
        Result of the operation as a formatted string
    """
    try:
        # Get Supabase credentials from environment variables
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            return "Error: Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables."
        
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Validate inputs
        if not table:
            return "Error: Table name is required."
        
        if operation_type not in ['insert', 'select', 'update', 'delete']:
            return "Error: Invalid operation type. Use 'insert', 'select', 'update', or 'delete'."
        
        # Perform the operation based on type
        if operation_type == 'insert':
            if not data:
                return "Error: Data is required for insert operation."
            
            result = supabase.table(table).insert(data).execute()
            return f"Insert successful: {len(result.data)} row(s) inserted into {table}"
        
        elif operation_type == 'select':
            query = supabase.table(table).select("*")
            
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            result = query.execute()
            return f"Select successful: Found {len(result.data)} row(s) in {table}\nData: {json.dumps(result.data, indent=2)}"
        
        elif operation_type == 'update':
            if not data or not filters:
                return "Error: Both data and filters are required for update operation."
            
            query = supabase.table(table).update(data)
            
            # Apply filters
            for key, value in filters.items():
                query = query.eq(key, value)
            
            result = query.execute()
            return f"Update successful: {len(result.data)} row(s) updated in {table}"
        
        elif operation_type == 'delete':
            if not filters:
                return "Error: Filters are required for delete operation."
            
            query = supabase.table(table).delete()
            
            # Apply filters
            for key, value in filters.items():
                query = query.eq(key, value)
            
            result = query.execute()
            return f"Delete successful: Rows deleted from {table}"
        
    except Exception as e:
        return f"Error performing Supabase operation: {str(e)}"

# System prompt to guide the model's behavior
#web_search: Search the google search engine when Tavily Search and Wikipedia Search do not return a result. Provide a specific search query.
#webpage_scrape: Scrape content from a specific webpage URL when Tavily Search and Wikipedia Search do not return a result. Provide a valid URL to extract information from a particular web page.
#Give preference to using Tavily Search and Wikipedia Search before using web_search or webpage_scrape. When Web_search does not return a result, use Tavily Search.

SYSTEM_PROMPT = """Answer the following questions as best you can. DO NOT rely on your internal knowledge unless web searches are rate-limited or you're specifically instructed to. You have access to the following tools:

python_code: Execute Python code. Provide the complete Python code as a string. Use this tool to calculate math problems.
wikipedia_search: Search Wikipedia for information about a specific topic. Optionally specify the number of results to return.
tavily_search: Search the web using Tavily for more comprehensive results. Optionally specify search_depth as 'basic' or 'comprehensive'.
arxiv_search: Search ArXiv for scientific papers on a specific topic. Optionally specify max_results to control the number of papers returned.
supabase_operation: Perform database operations on Supabase (insert, select, update, delete). Provide operation_type, table name, and optional data/filters.

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
python_code: Execute Python code, args: {"code": {"type": "string"}}
wikipedia_search: Search Wikipedia, args: {"query": {"type": "string"}, "num_results": {"type": "integer", "optional": true}}
tavily_search: Search with Tavily, args: {"query": {"type": "string"}, "search_depth": {"type": "string", "optional": true}}
arxiv_search: Search ArXiv papers, args: {"query": {"type": "string"}, "max_results": {"type": "integer", "optional": true}}
webpage_scrape: Scrape a specific webpage, args: {"url": {"type": "string"}}
supabase_operation: Perform database operations, args: {"operation_type": {"type": "string"}, "table": {"type": "string"}, "data": {"type": "object", "optional": true}, "filters": {"type": "object", "optional": true}}

IMPORTANT: Make sure your JSON is properly formatted with double quotes around keys and string values.

Example use for Supabase:

Insert data:
```json
{
  "action": "supabase_operation",
  "action_input": {"operation_type": "insert", "table": "users", "data": {"name": "John Doe", "email": "john@example.com"}}
}
```

Select data:
```json
{
  "action": "supabase_operation", 
  "action_input": {"operation_type": "select", "table": "users", "filters": {"id": 1}}
}
```

Update data:
```json
{
  "action": "supabase_operation",
  "action_input": {"operation_type": "update", "table": "users", "data": {"name": "Jane Doe"}, "filters": {"id": 1}}
}
```

Delete data:
```json
{
  "action": "supabase_operation",
  "action_input": {"operation_type": "delete", "table": "users", "filters": {"id": 1}}
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

NEVER fake or simulate tool output yourself.

... (this Thought/Action/Observation cycle can repeat as needed) ...
Thought: I now know the final answer
Final Answer: Directly answer the question in the shortest possible way. For example, if the question is "What is the capital of France?", the answer should be "Paris" without any additional text. If the question is "What is the population of New York City?", the answer should be "8.4 million" without any additional text.
Make sure to follow any formatting instructions given by the user.
Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer."""

# Generate the chat interface, including the tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
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

# Router function to direct to the correct tool
def router(state: AgentState) -> str:
    """Route to the appropriate tool based on the current_tool field."""
    tool = state.get("current_tool")
    action_input = state.get("action_input")
    print(f"Routing to: {tool}")
    print(f"Router received action_input: {action_input}")
    
    # if tool == "web_search":
    #     return "web_search"
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
    else:
        return "end"

# Create the graph
def create_agent_graph() -> StateGraph:
    """Create the complete agent graph with individual tool nodes."""
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    # builder.add_node("web_search", web_search_node)
    builder.add_node("python_code", python_code_node)
    builder.add_node("webpage_scrape", webpage_scrape_node)
    builder.add_node("wikipedia_search", wikipedia_search_node)
    builder.add_node("tavily_search", tavily_search_node)
    builder.add_node("arxiv_search", arxiv_search_node)
    builder.add_node("supabase_operation", supabase_operation_node)

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
            # "web_search": "web_search",
            "python_code": "python_code",
            "webpage_scrape": "webpage_scrape",
            "wikipedia_search": "wikipedia_search",
            "tavily_search": "tavily_search",
            "arxiv_search": "arxiv_search",
            "supabase_operation": "supabase_operation",
            "end": END
        }
    )
    
    # Tools always go back to assistant
    # builder.add_edge("web_search", "assistant")
    builder.add_edge("python_code", "assistant")
    builder.add_edge("webpage_scrape", "assistant")
    builder.add_edge("wikipedia_search", "assistant")
    builder.add_edge("tavily_search", "assistant")
    builder.add_edge("arxiv_search", "assistant")
    builder.add_edge("supabase_operation", "assistant")
    
    # Compile the graph
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
            result = self.graph.invoke(initial_state, {"recursion_limit": 100})
            
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
    response = agent("""Given this table defining * on the set S = {a, b, c, d, e}

|*|a|b|c|d|e|
|---|---|---|---|---|---|
|a|a|b|c|b|d|
|b|b|c|a|e|c|
|c|c|a|b|b|a|
|d|b|e|b|e|d|
|e|d|b|a|d|c|

provide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order.""")
    print("\nFinal Response:")
    print(response)

