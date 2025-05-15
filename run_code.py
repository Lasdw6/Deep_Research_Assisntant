#!/usr/bin/env python
import sys

def run_code(code_string):
    """Run the provided code string directly"""
    # Create a clean globals dictionary
    globals_dict = {
        '__name__': '__main__'
    }
    
    # Execute the code
    try:
        exec(code_string, globals_dict)
        print("Code executed successfully.")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Get code from command line argument or stdin
    if len(sys.argv) > 1:
        # Code is provided in the argument
        code = sys.argv[1]
    else:
        # Read code from stdin
        code = sys.stdin.read()
    
    # Run the code
    run_code(code) 
