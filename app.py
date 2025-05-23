import os
import gradio as gr
import requests
import inspect
import pandas as pd
import base64
from agent import ScholarAI
from rate_limiter import QueryRateLimiter
from flask import request
import PyPDF2
import fitz  # PyMuPDF
import time
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Load custom CSS
with open("static/custom.css", "r", encoding="utf-8") as f:
    custom_css = f.read()

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
ALLOWED_FILE_EXTENSIONS = [".mp3", ".xlsx", ".py", ".png", ".jpg", ".jpeg", ".gif", ".txt", ".md", ".json", ".csv", ".yml", ".yaml", ".html", ".css", ".js"]
MAX_FILE_SIZE_MB = 10
CHUNK_SIZE = 1000  # characters per chunk for text processing

# Initialize rate limiter (5 queries per hour)
query_limiter = QueryRateLimiter(max_queries_per_hour=5)

# Dictionary to store session-specific conversation histories
session_histories = {}

# --- Model Settings ---
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000
DEFAULT_MODEL = "gpt-4o-mini"

# --- Chat Interface Functions ---
def format_history_for_agent(history: list) -> str:
    """
    Format the chat history into a string that the agent can understand.
    """
    if not history:
        return ""
        
    formatted_history = []
    for message in history:
        if isinstance(message, dict) and "role" in message and "content" in message:
            role = message["role"]
            content = message["content"]
            formatted_history.append(f"{role.upper()}: {content}")
    
    return "\n".join(formatted_history)

def validate_inputs(question: str, file_uploads: List[gr.File]) -> Tuple[bool, str]:
    """Validate user inputs before processing."""
    if not question.strip() and (not file_uploads or len(file_uploads) == 0):
        return False, "Please enter a question or upload a file."
    
    if len(question) > 2000:
        return False, "Question is too long. Please keep it under 2000 characters."
    
    if file_uploads:
        for file in file_uploads:
            if file is None:
                continue
                
            file_path = file.name
            if not os.path.exists(file_path):
                return False, f"File {os.path.basename(file_path)} not found."
                
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            if file_size > MAX_FILE_SIZE_MB:
                return False, f"File {os.path.basename(file_path)} is too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
            
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ALLOWED_FILE_EXTENSIONS:
                return False, f"File {os.path.basename(file_path)} has an unsupported format. Allowed formats: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
    
    return True, ""

def process_document(file_path: str, progress=gr.Progress()) -> List[str]:
    """Process document and return chunks with progress bar."""
    file_ext = os.path.splitext(file_path)[1].lower()
    chunks = []
    
    try:
        if file_ext == '.pdf':
            # Process PDF
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            for page_num in progress.tqdm(range(total_pages), desc="Processing PDF pages"):
                page = doc[page_num]
                text = page.get_text()
                # Split text into chunks
                for i in range(0, len(text), CHUNK_SIZE):
                    chunk = text[i:i + CHUNK_SIZE]
                    if chunk.strip():
                        chunks.append(f"[Page {page_num + 1}] {chunk}")
                time.sleep(0.1)  # Small delay to show progress
                
        elif file_ext in ['.txt', '.md', '.json', '.csv', '.yml', '.yaml', '.html', '.css', '.js', '.py']:
            # Process text files
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                total_chunks = len(text) // CHUNK_SIZE + (1 if len(text) % CHUNK_SIZE else 0)
                
                for i in progress.tqdm(range(0, len(text), CHUNK_SIZE), desc="Processing text chunks"):
                    chunk = text[i:i + CHUNK_SIZE]
                    if chunk.strip():
                        chunks.append(chunk)
                    time.sleep(0.1)  # Small delay to show progress
                    
        elif file_ext in ['.xlsx']:
            # Process Excel files
            df = pd.read_excel(file_path)
            total_rows = len(df)
            
            for i in progress.tqdm(range(0, total_rows, CHUNK_SIZE), desc="Processing Excel rows"):
                chunk_df = df.iloc[i:i + CHUNK_SIZE]
                chunks.append(chunk_df.to_string())
                time.sleep(0.1)  # Small delay to show progress
                
        return chunks
        
    except Exception as e:
        return [f"Error processing file: {str(e)}"]

def chat_with_agent(question: str, file_uploads, history: list, temperature: float, max_tokens: int, model: str, progress=gr.Progress()) -> tuple:
    """
    Handle chat interaction with ScholarAI agent, now with file upload support and input validation.
    """
    # Validate inputs
    is_valid, error_message = validate_inputs(question, file_uploads)
    if not is_valid:
        history.append({"role": "assistant", "content": f"❌ {error_message}"})
        return history, ""
    
    try:
        # Use the history object's ID as a session identifier
        session_id = str(id(history))
        print(f"Using session ID: {session_id}")
        
        # Initialize or get session history
        if session_id not in session_histories:
            session_histories[session_id] = []
            if history:
                session_histories[session_id].extend(history)
        
        # Add the question to both histories immediately
        history.append({"role": "user", "content": question})
        session_histories[session_id].append({"role": "user", "content": question})
        
        try:
            # Initialize agent with current settings
            agent = ScholarAI(
                max_iterations=35,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model
            )
            print("Agent initialized successfully with Temperature: ", temperature, "Max Tokens: ", max_tokens, "Model: ", model)
        except ValueError as e:
            error_message = str(e)
            if "API key not found" in error_message:
                error_message = "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            elif "Invalid OpenAI API key" in error_message:
                error_message = "Invalid OpenAI API key. Please check your API key and try again."
            elif "rate limit" in error_message.lower() or "quota" in error_message.lower():
                error_message = "OpenAI API rate limit exceeded or quota reached. Please try again later."
            else:
                error_message = f"Error initializing AI agent: {error_message}"
            
            history.append({"role": "assistant", "content": error_message})
            if session_id in session_histories:
                session_histories[session_id].append({"role": "assistant", "content": error_message})
            return history, ""
        
        # Process uploaded files if any
        attachments = {}
        file_info = ""
        
        if file_uploads:
            for file in file_uploads:
                if file is not None:
                    file_path = file.name
                    file_name = os.path.basename(file_path)
                    
                    # Process document and get chunks
                    chunks = process_document(file_path, progress)
                    
                    if len(chunks) > 1:
                        file_info += f"\nProcessing {file_name} in {len(chunks)} chunks..."
                        
                        # Process each chunk
                        for i, chunk in enumerate(chunks, 1):
                            chunk_name = f"{file_name}_chunk_{i}"
                            chunk_content = base64.b64encode(chunk.encode('utf-8')).decode('utf-8')
                            attachments[chunk_name] = chunk_content
                            file_info += f"\nProcessed chunk {i}/{len(chunks)}"
                    else:
                        # Single chunk or error
                        with open(file_path, "rb") as f:
                            file_content = f.read()
                            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
                            attachments[file_name] = file_content_b64
                            file_info += f"\nUploaded file: {file_name}"
            
            if file_info:
                if question.strip():
                    question = f"{question}\n{file_info}"
                else:
                    question = f"Please analyze these files: {file_info}"
        
        # Format the session-specific conversation history
        conversation_history = format_history_for_agent(session_histories[session_id])
        
        # Prepare the full context for the agent
        full_context = f"Question: {question}\n\nConversation History:\n{conversation_history}"
        
        # Get response from agent with attachments if available
        if attachments:
            response = agent(full_context, attachments)
        else:
            response = agent(full_context)
        
        # Format the response to show thought process
        formatted_response = ""
        if "Thought:" in response:
            sections = response.split("\n\n")
            for section in sections:
                if section.startswith("Thought:"):
                    formatted_response += f"{section[7:].strip()}\n\n"
                elif section.startswith("Action:"):
                    if "action" in section and "action_input" in section:
                        try:
                            import json
                            action_json = json.loads(section.split("```json")[1].split("```")[0].strip())
                            tool_name = action_json.get("action", "").replace("_", " ").title()
                            formatted_response += f"Using {tool_name}...\n\n"
                        except:
                            formatted_response += f"{section[7:].strip()}\n\n"
                elif section.startswith("Observation:"):
                    formatted_response += f"{section[11:].strip()}\n\n"
                elif section.startswith("Final Answer:"):
                    formatted_response += f"{section[12:].strip()}\n\n"
                else:
                    formatted_response += f"{section}\n\n"
        else:
            formatted_response = response
        
        # Add response to both histories
        history.append({"role": "assistant", "content": formatted_response})
        session_histories[session_id].append({"role": "assistant", "content": formatted_response})
        
        return history, ""
        
    except Exception as e:
        error_str = str(e).lower()
        if "credit" in error_str or "quota" in error_str or "limit" in error_str or "exceeded" in error_str:
            error_message = "It seems I've run out of API credits. Please try again later or tomorrow when the credits reset."
        elif "invalid_api_key" in error_str or "incorrect_api_key" in error_str:
            error_message = "Invalid OpenAI API key. Please check your API key and try again."
        elif "api_key" in error_str:
            error_message = "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        else:
            error_message = f"Error: {str(e)}"
        
        history.append({"role": "assistant", "content": error_message})
        if session_id in session_histories:
            session_histories[session_id].append({"role": "assistant", "content": error_message})
        return history, ""

def clear_chat():
    """Clear the chat history."""
    return [], ""

# --- Evaluation Functions ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the ScholarAI on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = ScholarAI(
            max_iterations=35,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            model=DEFAULT_MODEL
        )
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []

    tasks = {"cca530fc-4052-43b2-b130-b30968d8aa44":"chess.png",
             "1f975693-876d-457b-a649-393859e79bf3":"audio1.mp3",
             "f918266a-b3e0-4914-865d-4faa564f1aef":"code.py",
             "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3":"audio2.mp3",
             "7bd855d8-463d-4ed5-93ca-5fe35145f733":"excel.xlsx"}
    file_path = "TEMPP/"
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            # Initialize question_text2 with the original question
            question_text2 = question_text
            
            # Add file path information if task_id is in tasks
            if task_id in tasks:
                question_text2 = question_text + f"\n\nThis is the file path: {file_path + tasks[task_id]}"
                
            # Get the answer from the agent
            submitted_answer = agent(question_text2)
            
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

# --- Build Gradio Interface using Blocks with Tabs ---
with gr.Blocks(title="ScholarAI Agent", css=custom_css) as demo:
    with gr.Row(elem_classes="header-bar"):
        with gr.Column(scale=3):
            gr.Markdown("# <span style='font-size: 2.5em'>ScholarAI</span>", elem_classes="title")
            gr.Markdown("""
            <div class="badges">
                <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status">
                <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
                <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="Version">
                <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python">
                <img src="https://img.shields.io/badge/gradio-5.29.1-orange" alt="Gradio">
            </div>
            """, elem_classes="badges-container")
        with gr.Column(scale=1):
            gr.Markdown("<span style='font-size: 0.9em'>by [Vividh Mahajan](https://huggingface.co/Lasdw)</span>", elem_classes="author")
    
    gr.Markdown("""
    ## ScholarAI helps you find answers by searching the web, analyzing images, processing audio, and more. 
    
    ### Tip: Ask specific, factual questions for best results. Some websites may be restricted.
    """)
    
    with gr.Accordion("Example Questions", open=False, elem_classes="example-questions"):
        gr.Markdown("""
        **Research & Analysis:**
        - "Find the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists. Tell me thier current age and where they are from."
        - "Analyze this image of a mathematical equation, and find an academic papers that use this equation."

        **Multi-Modal Analysis:**
        - "I have an interview recording and a transcript. Compare the audio transcription with the provided transcript, identify any discrepancies."
        - "This image shows a historical document. Find me the historical events from that era."

        **Code & Data Processing:**
        - "I have a Python script and an Excel file with data. Analyze the code's functionality and suggest improvements based on the data patterns."
        - "This code contains a bug. Debug it."

        The agent can handle multiple file uploads and combine information from various sources to provide comprehensive answers. Try asking complex questions that require multiple tools working together!
        """)
    
    with gr.Row():
        # Left panel - Chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=250,
                type="messages"
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g. Analyze this interview transcript and find discrepancies",
                    lines=5,
                    max_lines=5,
                    container=True,
                    scale=2,
                    min_width=500
                )
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_upload = gr.File(
                                label="Upload Files (.png, .txt, .mp3, .xlsx, .py)",
                                file_types=ALLOWED_FILE_EXTENSIONS,
                                file_count="multiple",
                                height=175,
                                min_width=200
                            )
            with gr.Row():
                submit_btn = gr.Button("Start Research", variant="primary")
        
        # Right panel - Controls
        with gr.Column(scale=1):
            gr.Markdown("# Model Settings")
            with gr.Group():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_TEMPERATURE,
                    step=0.1,
                    label="Temperature",
                    info="Higher values increase randomness, lower values increase determinism"
                )
                max_tokens = gr.Slider(
                    minimum=100,
                    maximum=4000,
                    value=DEFAULT_MAX_TOKENS,
                    step=100,
                    label="Max Tokens",
                    info="Maximum length of the response"
                )
                model = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-3.5-turbo"],
                    value=DEFAULT_MODEL,
                    label="Model",
                    info="The language model to use"
                )
    
    # Footer with disclaimer
    gr.Markdown("""
    <div class="footer">
    This tool is designed for educational and research purposes only. It is not intended for malicious use.
    </div>
    """)
    
    # Chat interface event handlers
    submit_btn.click(
        fn=chat_with_agent,
        inputs=[question_input, file_upload, chatbot, temperature, max_tokens, model],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=chat_with_agent,
        inputs=[question_input, file_upload, chatbot, temperature, max_tokens, model],
        outputs=[chatbot, question_input]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for ScholarAI Agent...")
    demo.launch(debug=True, share=False, show_api=False, favicon_path="static/favicon.ico", enable_monitoring=True)