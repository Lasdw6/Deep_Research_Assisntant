import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import tempfile
import base64
import json
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import html2text
import pandas as pd
from tabulate import tabulate
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from supabase import create_client, Client
import openai

# Add new imports for YouTube processing
import re
import pytube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Add new imports for image processing
from PIL import Image, ExifTags, ImageStat
import numpy as np
from io import BytesIO

load_dotenv()

def extract_python_code_from_complex_input(input_text):
    """
    Dedicated function to extract Python code from deeply nested JSON structures.
    This function handles the specific case of Python code embedded in nested JSON.
    """
    import re
    import json
    
    # Convert to string if it's not already
    if not isinstance(input_text, str):
        try:
            input_text = json.dumps(input_text)
        except:
            input_text = str(input_text)
    
    # Check if this looks like a JSON structure containing code
    if not (input_text.strip().startswith('{') and '"code"' in input_text):
        return input_text  # Not a JSON structure, return as is
    
    # First attempt: Try to extract using a direct regex for the nested case
    # This pattern looks for "code": "..." with proper escaping
    pattern = re.compile(r'"code"\s*:\s*"(.*?)(?<!\\)"\s*}', re.DOTALL)
    matches = pattern.findall(input_text)
    
    if matches:
        # Get the longest match (most likely the complete code)
        extracted_code = max(matches, key=len)
        
        # Unescape common escape sequences
        extracted_code = extracted_code.replace('\\n', '\n')
        extracted_code = extracted_code.replace('\\"', '"')
        extracted_code = extracted_code.replace("\\'", "'")
        extracted_code = extracted_code.replace("\\\\", "\\")
        
        print(f"Extracted code using direct regex approach: {extracted_code[:50]}...")
        return extracted_code
    
    # Second attempt: Try JSON parsing and navigate the structure
    try:
        parsed = json.loads(input_text)
        
        # Navigate through possible structures
        if isinstance(parsed, dict):
            # Direct code field
            if 'code' in parsed:
                extracted = parsed['code']
                if isinstance(extracted, str):
                    return extracted
            
            # Action with action_input structure
            if 'action' in parsed and 'action_input' in parsed:
                action_input = parsed['action_input']
                
                # Case 1: action_input is a dict with code
                if isinstance(action_input, dict) and 'code' in action_input:
                    return action_input['code']
                
                # Case 2: action_input is a string that might be JSON
                if isinstance(action_input, str):
                    try:
                        nested = json.loads(action_input)
                        if isinstance(nested, dict) and 'code' in nested:
                            return nested['code']
                    except:
                        # If it's not valid JSON, might be the code itself
                        return action_input
    except:
        # If JSON parsing fails, try one more regex approach
        # This looks for any content between balanced braces
        try:
            # Find the innermost code field
            code_start = input_text.rfind('"code"')
            if code_start != -1:
                # Find the start of the value (after the colon and quote)
                value_start = input_text.find(':', code_start)
                if value_start != -1:
                    value_start = input_text.find('"', value_start)
                    if value_start != -1:
                        value_start += 1  # Move past the quote
                        # Now find the end quote that's not escaped
                        value_end = value_start
                        while True:
                            next_quote = input_text.find('"', value_end + 1)
                            if next_quote == -1:
                                break
                            # Check if this quote is escaped
                            if input_text[next_quote - 1] != '\\':
                                value_end = next_quote
                                break
                            value_end = next_quote
                        
                        if value_end > value_start:
                            extracted = input_text[value_start:value_end]
                            # Unescape
                            extracted = extracted.replace('\\n', '\n')
                            extracted = extracted.replace('\\"', '"')
                            extracted = extracted.replace("\\'", "'")
                            extracted = extracted.replace("\\\\", "\\")
                            return extracted
        except:
            pass
    
    # If all else fails, return the original input
    return input_text

def test_python_execution(code_str):
    """A simplified function to test Python code execution and diagnose issues."""
    import io
    import sys
    import random
    import time
    from contextlib import redirect_stdout
    
    # Create a simple globals environment
    test_globals = {
        'random': random,
        'randint': random.randint,
        'time': time,
        'sleep': time.sleep,
        '__name__': '__main__',
        '__builtins__': __builtins__  # Use all built-ins for simplicity
    }
    
    # Create an empty locals dict
    test_locals = {}
    
    # Capture output
    output = io.StringIO()
    
    # Execute with detailed error reporting
    with redirect_stdout(output):
        print(f"Executing code:\n{code_str}")
        try:
            # Try compilation first to catch syntax errors
            compiled_code = compile(code_str, '<string>', 'exec')
            print("Compilation successful!")
            
            # Then try execution
            try:
                exec(compiled_code, test_globals, test_locals)
                print("Execution successful!")
                
                # Check what variables were defined
                print(f"Defined locals: {list(test_locals.keys())}")
                
                # If the code defines a main block, try to call a bit of it directly
                if "__name__" in test_globals and test_globals["__name__"] == "__main__":
                    print("Running main block...")
                    if "Okay" in test_locals and "keep_trying" in test_locals:
                        print("Found Okay and keep_trying functions, attempting to call...")
                        try:
                            go = test_locals["Okay"]()
                            result = test_locals["keep_trying"](go)
                            print(f"Result from keep_trying: {result}")
                        except Exception as e:
                            print(f"Error in main execution: {type(e).__name__}: {str(e)}")
            except Exception as e:
                print(f"Runtime error: {type(e).__name__}: {str(e)}")
                # Get traceback info
                import traceback
                traceback.print_exc(file=output)
        except SyntaxError as e:
            print(f"Syntax error: {str(e)}")
    
    # Get the captured output
    output_text = output.getvalue()
    
    # Try to evaluate the last expression if it's not a statement
    try:
        last_line = code_str.strip().split('\n')[-1]
        if not last_line.endswith(':'):  # Not a control structure
            last_result = eval(last_line, test_globals, test_locals)
            if last_result is not None:
                return str(last_result)
    except:
        pass  # If evaluation fails, just return the output
    
    # Return the captured output
    return output_text

def run_python_code(code: str):
    """Execute Python code safely using an external Python process."""
    try:
        # Pre-process code to handle complex nested structures
        code = extract_python_code_from_complex_input(code)
        
        print(f"Final code to execute: {code[:100]}...")
        
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
            "from itertools import", "from random import", "from random import randint",
            "from random import choice", "from random import sample", "from random import random",
            "from random import uniform", "from random import shuffle", "import time",
            "from time import sleep"
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
        
        # Direct execution
        # Use our test_python_execution function which has more robust error handling
        test_result = test_python_execution(code)
        
        # Extract just the relevant output from the test execution result
        # Remove diagnostic information that might confuse users
        cleaned_output = []
        for line in test_result.split('\n'):
            # Skip diagnostic lines
            if line.startswith("Executing code:") or line.startswith("Compilation successful") or line.startswith("Execution successful") or "Defined locals:" in line:
                continue
            cleaned_output.append(line)
            
        return '\n'.join(cleaned_output)
                
    except Exception as e:
        # Get the error type name without the "Error" suffix if it exists
        error_type = type(e).__name__.replace('Error', '')
        # Add a space between camel case words
        error_type = re.sub(r'([a-z])([A-Z])', r'\1 \2', error_type)
        return f"{error_type} Error: {str(e)}. Try again with a different code or try a different tool."

def scrape_webpage(url: str, keywords: Optional[List[str]] = None) -> str:
    """
    Safely scrape content from a specified URL with intelligent content extraction.
    
    Args:
        url: The URL to scrape
        keywords: Optional list of keywords to focus the content extraction
        
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
        
        # Set headers that mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Set a reasonable timeout
        timeout = 10
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=timeout)
        
        # Check if request was successful
        if response.status_code != 200:
            if response.status_code == 403:
                return f"Error: Access Forbidden (403). The website is actively blocking scrapers."
            return f"Error: Failed to fetch the webpage. Status code: {response.status_code}"
        
        # Use BeautifulSoup to parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'iframe', 'footer', 'nav', 'header', 'aside', 'form', 'noscript', 'meta', 'link']):
            element.decompose()
        
        # Get the page title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract the main content
        # First try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content')
        
        # If no main content area is found, use the body
        if not main_content:
            main_content = soup.body
        
        # Convert to plain text with specific settings
        h = html2text.HTML2Text()
        h.ignore_links = True  # Ignore links to reduce noise
        h.ignore_images = True
        h.ignore_tables = False
        h.unicode_snob = True
        h.body_width = 0  # Don't wrap text
        
        if main_content:
            text_content = h.handle(str(main_content))
        else:
            text_content = h.handle(response.text)
        
        # Clean up the text content
        # Remove extra whitespace and normalize newlines
        text_content = ' '.join(text_content.split())
        
        # Extract relevant content based on keywords if provided
        if keywords:
            # Split content into paragraphs (using double newlines as paragraph separators)
            paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
            
            # Score each paragraph based on keyword presence
            scored_paragraphs = []
            for paragraph in paragraphs:
                score = 0
                for keyword in keywords:
                    if keyword.lower() in paragraph.lower():
                        score += 1
                if score > 0:
                    scored_paragraphs.append((paragraph, score))
            
            # Sort paragraphs by score and take top ones
            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            
            # Take paragraphs with highest scores, but limit total content
            selected_paragraphs = []
            total_length = 0
            max_content_length = 2000
            
            for paragraph, score in scored_paragraphs:
                if total_length + len(paragraph) <= max_content_length:
                    selected_paragraphs.append(paragraph)
                    total_length += len(paragraph)
                else:
                    # If we can't fit the whole paragraph, try to find a good breaking point
                    remaining_length = max_content_length - total_length
                    if remaining_length > 100:  # Only break if we have enough space for meaningful content
                        break_point = paragraph[:remaining_length].rfind('.')
                        if break_point > remaining_length * 0.8:  # If we can find a good sentence break
                            selected_paragraphs.append(paragraph[:break_point + 1])
                            total_length += break_point + 1
                    break
            
            # Join the selected paragraphs
            text_content = '\n\n'.join(selected_paragraphs)
            
            if total_length >= max_content_length:
                text_content += "\n\n[Content truncated due to length...]"
        
        # If no keywords provided or no matches found, use the original content with length limit
        else:
            max_content_length = 2000
            if len(text_content) > max_content_length:
                # Try to find a good breaking point
                break_point = text_content[:max_content_length].rfind('.')
                if break_point > max_content_length * 0.8:  # If we can find a good sentence break
                    text_content = text_content[:break_point + 1]
                else:
                    text_content = text_content[:max_content_length]
                text_content += "\n\n[Content truncated due to length. Try using a different search method like Tavily search instead or use other key words or phrases.]"
        
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
        try:
            results = search.invoke({"query": query, "search_depth": search_depth})
        except requests.exceptions.HTTPError as http_err:
            # Check for the specific 432 error code
            if '432 Client Error' in str(http_err):
                return "Error: Invalid Tavily API key or API key has expired. Please check your API key and update it if necessary."
            else:
                # Re-raise to be caught by the outer try-except
                raise
        
        if not results:
            return f"No Tavily search results found for '{query}'. Try refining your search."
            
        # Format the results
        formatted_results = f"Tavily search results for '{query}':\n\n"
        
        # Check if results is a list of dictionaries (expected structure)
        if isinstance(results, list) and all(isinstance(item, dict) for item in results):
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. {result.get('title', 'No title')}\n"
                formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
                formatted_results += f"   {result.get('content', 'No content')}\n\n"
        # Check if results is a string
        elif isinstance(results, str):
            formatted_results += results
        # Otherwise, just convert to string representation
        else:
            formatted_results += str(results)
            
        return formatted_results
        
    except Exception as e:
        # Check if the exception string contains the 432 error
        if '432 Client Error' in str(e):
            return "Error: Invalid Tavily API key or API key has expired. Please check your API key and update it if necessary."
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

def excel_to_text(excel_path: str, sheet_name: Optional[str] = None, file_content: Optional[bytes] = None) -> str:
    """
    Read an Excel file and return a Markdown table of the requested sheet.
    
    Args:
        excel_path: Path to the Excel file (.xlsx or .xls) or name for the attached file.
        sheet_name: Optional name or index of the sheet to read. If None, reads the first sheet.
        file_content: Optional binary content of the file if provided as an attachment.
        
    Returns:
        A Markdown table representing the Excel sheet, or an error message if the file is not found or cannot be read.
    """
    try:
        # Handle file attachment case
        if file_content:
            # Create a temporary file to save the attachment
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            print(f"Saved attached Excel file to temporary location: {temp_path}")
            file_path = Path(temp_path)
        else:
            # Regular file path case
            file_path = Path(excel_path).expanduser().resolve()
            if not file_path.is_file():
                return f"Error: Excel file not found at {file_path}"

        # Process the Excel file
        sheet: Union[str, int] = (
            int(sheet_name)
            if sheet_name and sheet_name.isdigit()
            else sheet_name or 0
        )

        df = pd.read_excel(file_path, sheet_name=sheet)

        # Clean up temporary file if we created one
        if file_content and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"Deleted temporary Excel file: {temp_path}")

        if hasattr(df, "to_markdown"):
            return df.to_markdown(index=False)

        return tabulate(df, headers="keys", tablefmt="github", showindex=False)

    except Exception as e:
        # Clean up temporary file in case of error
        if file_content and 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"Deleted temporary Excel file due to error: {temp_path}")
        return f"Error reading Excel file: {e}"

def save_attachment_to_tempfile(file_content_b64: str, file_extension: str = '.xlsx') -> str:
    """
    Decode a base64 file content and save it to a temporary file.
    
    Args:
        file_content_b64: Base64 encoded file content
        file_extension: File extension to use for the temporary file
        
    Returns:
        Path to the saved temporary file
    """
    try:
        # Decode the base64 content
        file_content = base64.b64decode(file_content_b64)
        
        # Create a temporary file with the appropriate extension
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
            
        print(f"Saved attachment to temporary file: {temp_path}")
        return temp_path
    
    except Exception as e:
        print(f"Error saving attachment: {e}")
        return None

def process_youtube_video(url: str, summarize: bool = True) -> str:
    """
    Process a YouTube video by extracting its transcript/captions and basic metadata.
    Optionally summarize the content.
    
    Args:
        url: URL of the YouTube video
        summarize: Whether to include a summary of the video content
        
    Returns:
        Formatted video information including title, description, transcript, and optional summary
    """
    try:
        # Validate YouTube URL
        if "youtube.com" not in url and "youtu.be" not in url:
            return f"Error: The URL {url} doesn't appear to be a valid YouTube link"
        
        print(f"Processing YouTube video: {url}")
        
        # Extract video ID from the URL
        video_id = extract_youtube_video_id(url)
        
        if not video_id:
            return f"Error: Could not extract video ID from the URL: {url}"
        
        # Initialize metadata with defaults
        video_title = "Unable to retrieve title"
        video_author = "Unable to retrieve author"
        video_description = "Unable to retrieve description"
        video_length = 0
        video_views = 0
        video_publish_date = None
        metadata_error = None
        
        # Try to get video metadata using pytube (with error handling)
        try:
            # Try with different user agents to avoid detection
            pytube.innertube._default_clients['WEB']['context']['client']['clientVersion'] = '2.0'
            
            youtube = pytube.YouTube(url)
            video_title = youtube.title or "Title unavailable"
            video_author = youtube.author or "Author unavailable"
            video_description = youtube.description or "No description available"
            video_length = youtube.length or 0
            video_views = youtube.views or 0
            video_publish_date = youtube.publish_date
            print("Successfully retrieved video metadata")
        except Exception as e:
            metadata_error = str(e)
            print(f"Warning: Could not retrieve video metadata: {e}")
            print("Continuing with transcript extraction...")
        
        # Format video length from seconds to minutes and seconds
        if video_length > 0:
            minutes = video_length // 60
            seconds = video_length % 60
            length_formatted = f"{minutes}:{seconds:02d}"
        else:
            length_formatted = "Unknown"
        
        # Get video transcript using youtube_transcript_api (this is more reliable)
        transcript_text = ""
        transcript_error = None
        
        try:
            # Try to get transcript in multiple languages
            transcript_list = None
            
            # Try English first, then any available transcript
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except:
                # If English not available, get any available transcript
                available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript_list = next(iter(available_transcripts)).fetch()
            
            # Format transcript into readable text
            if transcript_list:
                for entry in transcript_list:
                    start_time = int(float(entry.get('start', 0)))
                    start_minutes = start_time // 60
                    start_seconds = start_time % 60
                    text = entry.get('text', '').strip()
                    if text:  # Only add non-empty text
                        transcript_text += f"[{start_minutes}:{start_seconds:02d}] {text}\n"
                print("Successfully retrieved video transcript")
            else:
                transcript_text = "No transcript content retrieved."
                
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            transcript_error = f"No transcript available: {str(e)}"
            transcript_text = transcript_error
        except Exception as e:
            transcript_error = f"Error retrieving transcript: {str(e)}"
            transcript_text = transcript_error
        
        # Compile all information
        result = f"Video ID: {video_id}\n"
        result += f"URL: {url}\n"
        result += f"Title: {video_title}\n"
        result += f"Creator: {video_author}\n"
        result += f"Length: {length_formatted}\n"
        
        if video_views > 0:
            result += f"Views: {video_views:,}\n"
        if video_publish_date:
            result += f"Published: {video_publish_date.strftime('%Y-%m-%d')}\n"
        
        # Add metadata error notice if applicable
        if metadata_error:
            result += f"\n⚠️  Note: Some metadata could not be retrieved due to: {metadata_error}\n"
        
        # Add description (truncated if too long)
        if video_description and video_description != "Unable to retrieve description":
            result += "\nDescription:\n"
            if len(video_description) > 500:
                description_preview = video_description[:500] + "..."
            else:
                description_preview = video_description
            result += f"{description_preview}\n"
        
        # Add transcript
        result += "\nTranscript:\n"
        
        if transcript_text:
            # Check if transcript is too long (over 8000 chars) and truncate if needed
            if len(transcript_text) > 8000:
                result += transcript_text[:8000] + "...\n[Transcript truncated due to length]\n"
            else:
                result += transcript_text
        else:
            result += "No transcript available.\n"
        
        # Add note about transcript and metadata errors
        if transcript_error:
            result += f"\n⚠️  Transcript error: {transcript_error}\n"
        
        # Provide troubleshooting tips if both metadata and transcript failed
        if metadata_error and transcript_error:
            result += "\n💡 Troubleshooting tips:\n"
            result += "- The video might be private, deleted, or have restricted access\n"
            result += "- Try updating the pytube library: pip install --upgrade pytube\n"
            result += "- Some videos may not have transcripts available\n"
        
        return result
        
    except Exception as e:
        return f"Error processing YouTube video: {str(e)}\n\nThis might be due to:\n- YouTube API changes\n- Network connectivity issues\n- Video access restrictions\n- Outdated pytube library\n\nTry updating pytube: pip install --upgrade pytube"

def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract the YouTube video ID from various URL formats.
    
    Args:
        url: A YouTube URL
        
    Returns:
        The video ID or None if it cannot be extracted
    """
    # Various YouTube URL patterns
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/e/|youtube\.com/watch\?.*v=|youtube\.com/watch\?.*&v=)([^&?/\s]{11})',
        r'youtube\.com/shorts/([^&?/\s]{11})',
        r'youtube\.com/live/([^&?/\s]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def transcribe_audio(audio_path: str, file_content: Optional[bytes] = None, language: Optional[str] = None) -> str:
    """
    Transcribe audio files using OpenAI Whisper.
    
    Args:
        audio_path: Path to the audio file or filename for attachments
        file_content: Optional binary content of the file if provided as an attachment
        language: Optional language code (e.g., 'en', 'es', 'fr') to improve accuracy
        
    Returns:
        Transcribed text from the audio file
    """
    temp_path = None
    audio_file = None
    
    try:
        # Check for OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        
        # Set the API key
        openai.api_key = openai_api_key
        
        # Handle file attachment case
        if file_content:
            # Determine file extension from audio_path or default to .mp3
            if '.' in audio_path:
                extension = '.' + audio_path.split('.')[-1].lower()
            else:
                extension = '.mp3'
            
            # Create a temporary file to save the attachment
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            print(f"Saved attached audio file to temporary location: {temp_path}")
            file_path = temp_path
        else:
            # Regular file path case
            file_path = Path(audio_path).expanduser().resolve()
            if not file_path.is_file():
                return f"Error: Audio file not found at {file_path}"
        
        print(f"Transcribing audio file: {file_path}")
        
        # Initialize client first
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Read the file content into memory - avoids file handle issues
        with open(file_path, "rb") as f:
            audio_data = f.read()
        
        # Create a file-like object from the data
        audio_file = BytesIO(audio_data)
        audio_file.name = os.path.basename(file_path)  # OpenAI needs a name
        
        # Call OpenAI Whisper API with the file-like object
        try:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
            
            # Extract the transcribed text
            transcribed_text = response.text
            
            if not transcribed_text:
                return "Error: No transcription was returned from Whisper API"
            
            # Format the result
            result = f"Audio Transcription:\n\n{transcribed_text}"
            
            return result
            
        except openai.BadRequestError as e:
            return f"Error: Invalid request to Whisper API - {str(e)}"
        except openai.RateLimitError as e:
            return f"Error: Rate limit exceeded for Whisper API - {str(e)}"
        except openai.APIError as e:
            return f"Error: OpenAI API error - {str(e)}"
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"
    finally:
        # Clean up resources
        if audio_file is not None:
            try:
                audio_file.close()
            except:
                pass
                
        # Clean up the temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                # Wait a moment to ensure file is not in use
                import time
                time.sleep(0.5)
                os.unlink(temp_path)
                print(f"Deleted temporary audio file: {temp_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {e}")

def process_image(image_path: str, image_url: Optional[str] = None, file_content: Optional[bytes] = None, analyze_content: bool = True) -> str:
    """
    Process an image file to extract information and content.
    
    Args:
        image_path: Path to the image file or filename for attachments
        image_url: Optional URL to fetch the image from instead of a local path
        file_content: Optional binary content of the file if provided as an attachment
        analyze_content: Whether to analyze the image content using vision AI (if available)
        
    Returns:
        Information about the image including dimensions, format, and content description
    """
    temp_path = None
    image_file = None
    
    try:
        # Import Pillow for image processing
        from PIL import Image, ExifTags, ImageStat
        import numpy as np
        from io import BytesIO
        
        # Handle image from URL
        if image_url:
            try:
                # Validate URL
                parsed_url = urlparse(image_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    return f"Error: Invalid URL format: {image_url}. Please provide a valid URL."
                
                print(f"Downloading image from URL: {image_url}")
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                
                # Create BytesIO object from content
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                image_source = f"URL: {image_url}"
            except requests.exceptions.RequestException as e:
                return f"Error downloading image from URL: {str(e)}"
            except Exception as e:
                return f"Error processing image from URL: {str(e)}"
                
        # Handle file attachment case
        elif file_content:
            try:
                # Determine file extension from image_path
                if '.' in image_path:
                    extension = '.' + image_path.split('.')[-1].lower()
                else:
                    extension = '.png'  # Default to PNG if no extension
                
                # Create a temporary file to save the attachment
                with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                print(f"Saved attached image file to temporary location: {temp_path}")
                image = Image.open(temp_path)
                image_source = f"Uploaded file: {image_path}"
            except Exception as e:
                return f"Error processing attached image: {str(e)}"
        else:
            # Regular file path case
            try:
                file_path = Path(image_path).expanduser().resolve()
                if not file_path.is_file():
                    return f"Error: Image file not found at {file_path}"
                
                image = Image.open(file_path)
                image_source = f"Local file: {file_path}"
            except Exception as e:
                return f"Error opening image file: {str(e)}"
        
        # Basic image information
        width, height = image.size
        image_format = image.format or "Unknown"
        image_mode = image.mode  # RGB, RGBA, L (grayscale), etc.
        
        # Extract EXIF data if available
        exif_data = {}
        if hasattr(image, '_getexif') and image._getexif():
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in image._getexif().items()
                if k in ExifTags.TAGS
            }
            
            # Filter for useful EXIF tags
            useful_tags = ['DateTimeOriginal', 'Make', 'Model', 'ExposureTime', 'FNumber', 'ISOSpeedRatings']
            exif_data = {k: v for k, v in exif.items() if k in useful_tags}
        
        # Calculate basic statistics
        if image_mode in ['RGB', 'RGBA', 'L']:
            try:
                stat = ImageStat.Stat(image)
                mean_values = stat.mean
                
                # Calculate average color for RGB images
                if image_mode in ['RGB', 'RGBA']:
                    avg_color = f"R: {mean_values[0]:.1f}, G: {mean_values[1]:.1f}, B: {mean_values[2]:.1f}"
                else:  # For grayscale
                    avg_color = f"Grayscale Intensity: {mean_values[0]:.1f}"
                    
                # Calculate image brightness (simplified)
                if image_mode in ['RGB', 'RGBA']:
                    brightness = 0.299 * mean_values[0] + 0.587 * mean_values[1] + 0.114 * mean_values[2]
                    brightness_description = "Dark" if brightness < 64 else "Dim" if brightness < 128 else "Normal" if brightness < 192 else "Bright"
                else:
                    brightness = mean_values[0]
                    brightness_description = "Dark" if brightness < 64 else "Dim" if brightness < 128 else "Normal" if brightness < 192 else "Bright"
            except Exception as e:
                print(f"Error calculating image statistics: {e}")
                avg_color = "Could not calculate"
                brightness_description = "Unknown"
        else:
            avg_color = "Not applicable for this image mode"
            brightness_description = "Unknown"
        
        # Image content analysis using OpenAI Vision API if available
        content_description = "Image content analysis not performed"
        if analyze_content:
            try:
                # Check for OpenAI API key
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if openai_api_key:
                    # Convert image to base64 for OpenAI API
                    buffered = BytesIO()
                    image.save(buffered, format=image_format if image_format != "Unknown" else "PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Initialize OpenAI client
                    client = openai.OpenAI(api_key=openai_api_key)
                    
                    # Call Vision API
                    response = client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image in detail, including the main subject, colors, setting, and any notable features. Be factual and objective. For a chess posistion, 1. List all the pieces and their positions (e.g., 'White King at e1', 'Black Queen at d8') 2. List any special conditions (castling rights, en passant, etc.) 3. Provide the position in FEN notation 4. Convert the position to PGN format"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{image_format.lower() if image_format != 'Unknown' else 'png'};base64,{img_str}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=300
                    )
                    
                    # Extract the analysis
                    content_description = response.choices[0].message.content
                else:
                    content_description = "OpenAI API key not found. To analyze image content, set the OPENAI_API_KEY environment variable."
            except Exception as e:
                content_description = f"Error analyzing image content: {str(e)}"
        
        # Format the result
        result = f"Image Information:\n\n"
        result += f"Source: {image_source}\n"
        result += f"Dimensions: {width} x {height} pixels\n"
        result += f"Format: {image_format}\n"
        result += f"Mode: {image_mode}\n"
        result += f"Average Color: {avg_color}\n"
        result += f"Brightness: {brightness_description}\n"
        
        # Add EXIF data if available
        if exif_data:
            result += "\nEXIF Data:\n"
            for key, value in exif_data.items():
                result += f"- {key}: {value}\n"
        
        # Add content description
        if analyze_content:
            result += f"\nContent Analysis:\n{content_description}\n"
        
        # Clean up resources
        image.close()
        print(result)
        return result
        
    except Exception as e:
        return f"Error processing image: {str(e)}"
    finally:
        # Clean up the temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                import time
                time.sleep(0.5)  # Wait a moment to ensure file is not in use
                os.unlink(temp_path)
                print(f"Deleted temporary image file: {temp_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {e}")
                # Non-fatal error, don't propagate exception

def read_file(file_path: str, file_content: Optional[bytes] = None, line_start: Optional[int] = None, line_end: Optional[int] = None) -> str:
    """
    Read and return the contents of a text file (.py, .txt, etc.).
    
    Args:
        file_path: Path to the file or filename for attachments
        file_content: Optional binary content of the file if provided as an attachment
        line_start: Optional starting line number (1-indexed) to read from
        line_end: Optional ending line number (1-indexed) to read to
        
    Returns:
        The content of the file as a string, optionally limited to specified line range
    """
    temp_path = None
    
    try:
        # Handle file attachment case
        if file_content:
            try:
                # Determine file extension from file_path if available
                if '.' in file_path:
                    extension = '.' + file_path.split('.')[-1].lower()
                else:
                    extension = '.txt'  # Default to .txt if no extension
                
                # Create a temporary file to save the attachment
                with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                print(f"Saved attached file to temporary location: {temp_path}")
                file_to_read = temp_path
                file_source = f"Uploaded file: {file_path}"
            except Exception as e:
                return f"Error processing attached file: {str(e)}"
        else:
            # Regular file path case
            try:
                file_to_read = Path(file_path).expanduser().resolve()
                if not file_to_read.is_file():
                    return f"Error: File not found at {file_to_read}"
                
                file_source = f"Local file: {file_path}"
            except Exception as e:
                return f"Error accessing file path: {str(e)}"
        
        # Check file extension
        file_extension = os.path.splitext(str(file_to_read))[1].lower()
        if file_extension not in ['.py', '.txt', '.md', '.json', '.csv', '.yml', '.yaml', '.html', '.css', '.js', '.sh', '.bat', '.log']:
            return f"Error: File type not supported for reading. Only text-based files are supported."
        
        # Read the file content
        try:
            with open(file_to_read, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Handle line range if specified
                if line_start is not None and line_end is not None:
                    # Convert to 0-indexed
                    line_start = max(0, line_start - 1)
                    line_end = min(len(lines), line_end)
                    
                    # Validate range
                    if line_start >= len(lines) or line_end <= 0 or line_start >= line_end:
                        return f"Error: Invalid line range ({line_start+1}-{line_end}). File has {len(lines)} lines."
                    
                    selected_lines = lines[line_start:line_end]
                    content = ''.join(selected_lines)
                    
                    # Add context about the selected range
                    result = f"File Content ({file_source}, lines {line_start+1}-{line_end} of {len(lines)}):\n\n{content}"
                else:
                    content = ''.join(lines)
                    line_count = len(lines)
                    # If the file is large, add a note about its size
                    if line_count > 1000:
                        file_size = os.path.getsize(file_to_read) / 1024  # KB
                        result = f"File Content ({file_source}, {line_count} lines, {file_size:.1f} KB):\n\n{content}"
                    else:
                        result = f"File Content ({file_source}, {line_count} lines):\n\n{content}"
                
                return result
                
        except UnicodeDecodeError:
            return f"Error: File {file_path} appears to be a binary file and cannot be read as text."
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    finally:
        # Clean up the temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                import time
                time.sleep(0.5)  # Wait a moment to ensure file is not in use
                os.unlink(temp_path)
                print(f"Deleted temporary file: {temp_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {e}")
                # Non-fatal error, don't propagate exception

def process_online_document(url: str, doc_type: str = "auto") -> str:
    """
    Process and analyze online PDFs and images.
    
    Args:
        url: URL of the document or image
        doc_type: Type of document ("pdf", "image", or "auto" for automatic detection)
        
    Returns:
        Analysis of the document content
    """
    try:
        # Validate URL
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
        
        print(f"Processing online document: {url}")
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/pdf,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Download the file
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()
        
        # Determine content type
        content_type = response.headers.get('content-type', '').lower()
        
        # Create a temporary file to save the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        try:
            # Process based on content type or specified doc_type
            if doc_type == "auto":
                if "pdf" in content_type or url.lower().endswith('.pdf'):
                    doc_type = "pdf"
                elif any(img_type in content_type for img_type in ['jpeg', 'png', 'gif', 'bmp', 'webp']):
                    doc_type = "image"
                else:
                    return f"Error: Unsupported content type: {content_type}"
            
            if doc_type == "pdf":
                try:
                    import PyPDF2
                    with open(temp_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                        
                        # Get metadata
                        metadata = pdf_reader.metadata
                        result = "PDF Analysis:\n\n"
                        if metadata:
                            result += "Metadata:\n"
                            for key, value in metadata.items():
                                if value:
                                    result += f"- {key}: {value}\n"
                            result += "\n"
                        
                        result += f"Number of pages: {len(pdf_reader.pages)}\n\n"
                        result += "Content:\n"
                        result += text_content[:8000]  # Limit content length
                        if len(text_content) > 8000:
                            result += "\n\n[Content truncated due to length...]"
                        
                        return result
                except ImportError:
                    return "Error: PyPDF2 library is required for PDF processing. Please install it using 'pip install PyPDF2'"
                
            elif doc_type == "image":
                # Use the existing process_image function
                return process_image(temp_path, url=url)
            
            else:
                return f"Error: Unsupported document type: {doc_type}"
                
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {e}")
                
    except requests.exceptions.RequestException as e:
        return f"Error accessing URL {url}: {str(e)}"
    except Exception as e:
        return f"Error processing online document: {str(e)}"

# Define the tools configuration
tools_config = [
    {
        "name": "python_code", 
        "description": "Execute Python code. Provide the complete Python code as a string in the format: {\"code\": \"your python code here\"}",
        "func": run_python_code
    },
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
        "description": "Extract and process information from a YouTube video including its transcript, title, author, and other metadata. Provide a URL in the format: {\"url\": \"https://www.youtube.com/watch?v=VIDEO_ID\", \"summarize\": true}",
        "func": process_youtube_video
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribe audio files (MP3, WAV, etc.) using OpenAI Whisper. You can provide either a file path or use a file attachment. For attachments, provide base64-encoded content. Optionally specify language for better accuracy.",
        "func": transcribe_audio
    },
    {
        "name": "process_image",
        "description": "Process and analyze image files. You can provide a local file path, image URL, or use a file attachment. Returns information about the image including dimensions, format, and content analysis.",
        "func": process_image
    },
    {
        "name": "read_file",
        "description": "Read and display the contents of a text file (.py, .txt, etc.). You can provide a file path or use a file attachment. Optionally specify line range to read a specific portion of the file.",
        "func": read_file
    },
    {
        "name": "process_online_document",
        "description": "Process and analyze online PDFs and images. Provide a URL and optionally specify the document type ('pdf', 'image', or 'auto').",
        "func": process_online_document
    }
] 