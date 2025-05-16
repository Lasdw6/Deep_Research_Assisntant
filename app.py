import os
import gradio as gr
import requests
import inspect
import pandas as pd
import base64
from agent import TurboNerd

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
ALLOWED_FILE_EXTENSIONS = [".mp3", ".xlsx", ".py", ".png", ".jpg", ".jpeg", ".gif", ".txt", ".md", ".json", ".csv", ".yml", ".yaml", ".html", ".css", ".js"]

# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.agent = TurboNerd()
        
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        answer = self.agent(question)
        print(f"Agent returning answer: {answer[:50]}...")
        return answer

# --- Chat Interface Functions ---
def chat_with_agent(question: str, file_uploads, history: list) -> tuple:
    """
    Handle chat interaction with TurboNerd agent, now with file upload support.
    """
    if not question.strip() and not file_uploads:
        return history, ""
    
    try:
        # Initialize agent
        agent = TurboNerd()
        
        # Process uploaded files if any
        attachments = {}
        file_info = ""
        
        if file_uploads:
            for file in file_uploads:
                if file is not None:
                    file_path = file.name
                    file_name = os.path.basename(file_path)
                    file_ext = os.path.splitext(file_name)[1].lower()
                    
                    # Check if file extension is allowed
                    if file_ext in ALLOWED_FILE_EXTENSIONS:
                        # Read file content and encode as base64
                        with open(file_path, "rb") as f:
                            file_content = f.read()
                            file_content_b64 = base64.b64encode(file_content).decode("utf-8")
                            attachments[file_name] = file_content_b64
                            file_info += f"\nUploaded file: {file_name}"
            
            if file_info:
                if question.strip():
                    question = question + file_info
                else:
                    question = f"Please analyze these files: {file_info}"
        
        # Get response from agent with attachments if available
        if attachments:
            response = agent(question, attachments)
        else:
            response = agent(question)
        
        # Format the response to show thought process
        formatted_response = ""
        if "Thought:" in response:
            # Split the response into sections
            sections = response.split("\n\n")
            for section in sections:
                if section.startswith("Thought:"):
                    formatted_response += f"🤔 {section[7:].strip()}\n\n"
                elif section.startswith("Action:"):
                    # Extract the tool being used
                    if "action" in section and "action_input" in section:
                        try:
                            import json
                            action_json = json.loads(section.split("```json")[1].split("```")[0].strip())
                            tool_name = action_json.get("action", "").replace("_", " ").title()
                            formatted_response += f"🛠️ Using {tool_name}...\n\n"
                        except:
                            formatted_response += f"🛠️ {section[7:].strip()}\n\n"
                elif section.startswith("Observation:"):
                    formatted_response += f"📝 {section[11:].strip()}\n\n"
                elif section.startswith("Final Answer:"):
                    formatted_response += f"✨ {section[12:].strip()}\n\n"
                else:
                    formatted_response += f"{section}\n\n"
        else:
            formatted_response = response
        
        # Add question and response to history in the correct format
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": formatted_response})
        
        return history, ""
    except Exception as e:
        error_message = f"Error: {str(e)}"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_message})
        return history, ""

def clear_chat():
    """Clear the chat history."""
    return [], "", None

# --- Evaluation Functions ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
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

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
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
with gr.Blocks(title="TurboNerd Agent🤓") as demo:
    gr.Markdown("# TurboNerd 🤓- The Deep Research Agent \n ### Made by Vividh Mahajan - @Lasdw on HuggingFace")
    
    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.TabItem("🤓", id="chat"):
            gr.Markdown("""
            ## Chat with TurboNerd 🤓
            Ask any question and get an answer from TurboNerd. The agent can search the web, Wikipedia, analyze images, process audio, and more!

            ### Example Questions:

            **Research & Analysis:**
            - "Find the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists. Cross-reference this information with their Wikipedia page and any recent news articles."
            - "Analyze this image of a mathematical equation, explain the concepts involved, and find similar problems from textbooks or academic papers."

            **Multi-Modal Analysis:**
            - "I have an interview recording and a transcript. Compare the audio transcription with the provided transcript, identify any discrepancies, and summarize the key points discussed."
            - "This image shows a historical document. Extract the text, identify the time period, and find related historical events from that era."

            **Code & Data Processing:**
            - "I have a Python script and an Excel file with data. Analyze the code's functionality, identify potential optimizations, and suggest improvements based on the data patterns."
            - "This code contains a bug. Debug it, explain the issue, and propose a solution. Then test the solution with the provided test cases."

            The agent can handle multiple file uploads and combine information from various sources to provide comprehensive answers. Try asking complex questions that require multiple tools working together!
            """)
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=600,
                        min_width=600,
                        max_width=1200
                    )
                    with gr.Row():
                        with gr.Column(scale=4):
                            msg = gr.Textbox(
                                placeholder="Ask me anything...",
                                show_label=False,
                                container=False
                            )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("Send", variant="primary")
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        retry_btn = gr.Button("Retry Last Response")
                        undo_btn = gr.Button("Undo Last Exchange")
            
            # Chat interface event handlers
            submit_btn.click(
                fn=chat_with_agent,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                fn=chat_with_agent,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, msg]
            )
        
        # Tab 2: Evaluation Interface
        with gr.TabItem(" ", id="evaluation"):
            gr.Markdown("""
            # You found a secret page 🤫
            ## Agent Evaluation Runner for the AI Agents course on HF :P
            ## See my ranking (@Lasdw) on the course [here](https://huggingface.co/spaces/agents-course/Students_leaderboard)
            
            ## Below is the original README.md for the space
                        
            **Instructions:**
            
            1. Log in to your Hugging Face account using the button below.
            2. Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
            
            ---
            **Disclaimers:**
            Once clicking on the "submit" button, it can take quite some time (this is the time for the agent to go through all the questions).
            This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution.
            """)
            
            gr.LoginButton()
            
            run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
            
            status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
            results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
            
            run_button.click(
                fn=run_and_submit_all,
                outputs=[status_output, results_table]
            )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for TurboNerd Agent...")
    demo.launch(debug=True, share=False)