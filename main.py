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
    DATA_DIR,
    DB_DIR
)

def setup_argparse():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI-powered candidate matching system"
    )
    
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Test the Gemini model connection"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Query for matching candidates",
        default=None
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode to query candidates"
    )
    
    return parser.parse_args()

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
    logger.info("Starting AI-Austin-Agent")
    
    # Parse command line arguments
    args = setup_argparse()
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Get the API keys from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return
    
    # Auto mode (default) - detect what needs to be done
    logger.info("Checking what needs to be done...")
    
    # Check if profile data exists
    has_profiles = check_profiles_data_exists()
    if not has_profiles:
        logger.info("Profile data not found. Starting data collection...")
        scrape_profiles()
    else:
        logger.info("Profile data already exists. Skipping collection.")
    
    # Check if vector database exists
    has_vector_db = check_vector_db_exists()
    if not has_vector_db:
        logger.info("Vector database not found. Creating database...")
        create_vector_db(gemini_api_key)
    else:
        logger.info("Vector database already exists. Skipping creation.")
    
    # Handle different modes
    if args.test_model:
        logger.info("Testing Gemini model connection...")
        test_gemini_model(gemini_api_key)
    elif args.query:
        logger.info(f"Processing query: {args.query}")
        result = query_candidates(args.query, gemini_api_key)
        print("\nMatching Candidates:\n")
        print(result)
    elif args.interactive:
        logger.info("Running in interactive mode")
        interactive_mode(gemini_api_key)
    else:
        # If no specific mode is selected, run in interactive mode
        logger.info("No mode specified, running in interactive mode")
        interactive_mode(gemini_api_key)

if __name__ == "__main__":
    main()
