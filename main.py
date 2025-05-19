import os
import argparse
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Import from consolidated ai_agent module
from ai_agent import (
    scrape_profiles, 
    create_vector_db, 
    query_candidates,
    check_profiles_data_exists,
    check_vector_db_exists,
    extract_experience_and_skills,
    DATA_DIR,
    DB_DIR
)

def setup_argparse():
    """Set up argument parser for the application"""
    parser = argparse.ArgumentParser(description="AI Austin Recruitment Agent")
    
    # Set up subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create parser for the 'recruiter' command
    recruiter_parser = subparsers.add_parser("recruiter", help="Run the recruiter agent")
    recruiter_parser.add_argument("--query", type=str, help="The recruitment query to process")
    
    # Create parser for the 'candidate' command
    candidate_parser = subparsers.add_parser("candidate", help="Run the candidate agent")
    candidate_parser.add_argument("--name", type=str, help="The candidate's name")

    # Add extract-profiles command
    extract_parser = subparsers.add_parser("extract-profiles", help="Extract experience and skills from HTML profiles")
    
    return parser

def test_gemini_model(api_key):
    """Test the connection to the Gemini model"""
    try:
        # Create a ChatGoogleGenerativeAI instance
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Example prompt
        prompt = "Tell me something interesting about AI recruitment"
        
        # Format as LangChain message
        messages = [
            HumanMessage(content=prompt)
        ]
        
        # Get response from the model
        logger.info("Sending test request to Gemini API...")
        response = gemini_llm.invoke(messages)
        
        logger.success("Gemini model test successful!")
        print("\nGemini says:")
        print(response.content)
        return True
    except Exception as e:
        logger.error(f"Error using Gemini: {e}")
        return False

def interactive_mode(api_key):
    """Run the application in interactive mode"""
    print("\n=== AI-Powered Candidate Matcher ===")
    print("Enter your job requirements to find matching candidates.")
    print("Type 'exit' to quit.")
    print("-----------------------------------")
    
    while True:
        query = input("\nEnter job requirements: ")
        
        if query.lower() in ('exit', 'quit', 'q'):
            print("Goodbye!")
            break
        
        if not query.strip():
            print("Please enter a valid query.")
            continue
        
        try:
            print("\nSearching for candidates...\n")
            result = query_candidates(query, api_key)
            print(result)
            print("\n-----------------------------------")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Sorry, an error occurred: {e}")

def main():
    """Main function to run the AI agent"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command == "recruiter":
        if not args.query:
            print("Please provide a recruitment query using --query")
            return
        
        print(f"Processing recruitment query: {args.query}")
        agent = AIAgent()
        result = agent.run_recruiter_agent(args.query)
        print(result)
        
    elif args.command == "candidate":
        if not args.name:
            print("Please provide a candidate name using --name")
            return
        
        print(f"Processing candidate: {args.name}")
        agent = AIAgent()
        result = agent.run_candidate_agent(args.name)
        print(result)
    
    elif args.command == "extract-profiles":
        print("Extracting experience and skills from HTML profiles...")
        profiles_updated = extract_experience_and_skills()
        print(f"Updated {profiles_updated} profiles with experience and skills data")
    
    else:
        print("Please specify a valid command. Use --help for more information.")

if __name__ == "__main__":
    main()
