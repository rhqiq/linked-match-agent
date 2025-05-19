import os
import argparse
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Import Pydantic models
from models import Profile, Experience, CandidateMatch, SearchResponse

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
    
    # Add example query command
    example_parser = subparsers.add_parser("example", help="Run an example query to test the system")
    
    # Add rebuild-db command
    rebuild_parser = subparsers.add_parser("rebuild-db", help="Rebuild the vector database with updated profile information")
    
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
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.")
        print("Please add it to your .env file or environment variables.")
        return
    
    if args.command == "recruiter":
        if not args.query:
            print("Please provide a recruitment query using --query")
            return
        
        print(f"Processing recruitment query: {args.query}")
        result = query_candidates(args.query, api_key)
        print(result)
        
    elif args.command == "candidate":
        if not args.name:
            print("Please provide a candidate name using --name")
            return
        
        print(f"Processing candidate: {args.name}")
        # Implementation for candidate lookup would go here
        print("Candidate lookup not yet implemented")
    
    elif args.command == "extract-profiles":
        print("Extracting experience and skills from HTML profiles...")
        profiles_updated = extract_experience_and_skills()
        print(f"Updated {profiles_updated} profiles with experience and skills data")
    
    elif args.command == "example":
        print("Running example query for a DevOps Engineer position...")
        example_query = """
        We are looking for a DevOps Engineer with the following qualifications:
        - 5+ years of experience in DevOps or related field
        - Strong experience with Kubernetes and container orchestration
        - Experience with CI/CD pipelines
        - Knowledge of infrastructure as code (Terraform, CloudFormation)
        - Experience with cloud platforms (AWS, Azure, or GCP)
        
        The candidate should have excellent communication skills and be able to work in a team environment.
        """
        
        # First check if we have profiles and a vector database
        has_profiles = check_profiles_data_exists()
        has_vector_db = check_vector_db_exists()
        
        if not has_profiles:
            print("Error: No profile data found. Please run 'extract-profiles' first.")
            return
        
        if not has_vector_db:
            print("Creating vector database...")
            create_vector_db(api_key)
        
        # Test retrieve one profile for debugging
        print("\nTesting profile retrieval...")
        from ai_agent import load_vector_db
        db = load_vector_db(api_key)
        docs = db.similarity_search("DevOps Kubernetes", k=1)
        if docs:
            print(f"Found profile: {docs[0].metadata.get('name', 'Unknown')}")
            print(f"Content preview (first 200 chars): {docs[0].page_content[:200]}...")
        
        print("\nQuerying for matching candidates...\n")
        result = query_candidates(example_query, api_key)
        print(result)
    
    elif args.command == "rebuild-db":
        print("Rebuilding vector database with updated profile information...")
        
        # Check for API key
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not found.")
            return
        
        # First check if we have profiles 
        has_profiles = check_profiles_data_exists()
        if not has_profiles:
            print("Error: No profile data found. Please run 'extract-profiles' first.")
            return
        
        # Check if database already exists and remove it
        import shutil
        if DB_DIR.exists():
            print(f"Removing existing database at {DB_DIR}...")
            shutil.rmtree(DB_DIR)
            print("Database removed.")
        
        # Create directory
        os.makedirs(DB_DIR, exist_ok=True)
        
        # Create new database
        print("Creating new vector database...")
        create_vector_db(api_key)
        print("Vector database rebuilt successfully!")
    
    else:
        # Default to interactive mode
        print("Starting interactive mode. Use --help for available commands.")
        interactive_mode(api_key)

if __name__ == "__main__":
    main()
