import os
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
        
        # Execute the entire code block at once
        with redirect_stdout(captured_output):
            # Try to evaluate as expression first (for simple expressions)
            lines = code.strip().split('\n')
            if len(lines) == 1 and not any(keyword in code for keyword in ['=', 'import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'with']):
                try:
                    result = eval(code, restricted_globals, local_scope)
                    print(f"Result: {result}")
                except:
                    # If eval fails, use exec
                    exec(code, restricted_globals, local_scope)
            else:
                # For multi-line code, execute the entire block
                exec(code, restricted_globals, local_scope)
        
        # Get the captured output
        output = captured_output.getvalue()
        
        if output.strip():
            return output.strip()
        else:
            # If no output, check if there's a result from the last expression
            lines = code.strip().split('\n')
            last_line = lines[-1].strip() if lines else ""
            
            # If the last line looks like an expression, try to evaluate it
            if last_line and not any(keyword in last_line for keyword in ['=', 'import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'with', 'print']):
                try:
                    result = eval(last_line, restricted_globals, local_scope)
                    return f"Result: {result}"
                except:
                    pass
                    
            return "Code executed successfully with no output."
                
    except SyntaxError as e:
        return f"Syntax Error: {str(e)}"
    except NameError as e:
        return f"Name Error: {str(e)}"
    except ZeroDivisionError as e:
        return f"Zero Division Error: {str(e)}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

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
    try:
        # Check for OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        
        # Set the API key
        openai.api_key = openai_api_key
        
        # Handle file attachment case
        temp_path = None
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
        
        # Open the audio file and transcribe using Whisper
        with open(file_path, "rb") as audio_file:
            # Prepare the transcription request
            transcript_params = {
                "model": "whisper-1",
                "file": audio_file
            }
            
            # Add language parameter if specified
            if language:
                transcript_params["language"] = language
            
            # Call OpenAI Whisper API
            response = openai.Audio.transcribe(**transcript_params)
            
            # Extract the transcribed text
            transcribed_text = response.get("text", "")
            
            if not transcribed_text:
                return "Error: No transcription was returned from Whisper API"
            
            # Clean up temporary file if we created one
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"Deleted temporary audio file: {temp_path}")
            
            # Format the result
            result = f"Audio Transcription:\n\n{transcribed_text}"
            
            # Add metadata if available
            if hasattr(response, 'duration'):
                result = f"Duration: {response.duration} seconds\n" + result
            
            return result
            
    except openai.error.InvalidRequestError as e:
        # Clean up temporary file in case of error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Error: Invalid request to Whisper API - {str(e)}"
    except openai.error.RateLimitError as e:
        # Clean up temporary file in case of error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Error: Rate limit exceeded for Whisper API - {str(e)}"
    except openai.error.APIError as e:
        # Clean up temporary file in case of error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Error: OpenAI API error - {str(e)}"
    except Exception as e:
        # Clean up temporary file in case of error
        if temp_path and 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Error transcribing audio: {str(e)}"

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
    }
] 