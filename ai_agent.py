import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create data directories if they don't exist
DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
DB_DIR = Path("chroma_db")
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Path to Excel file and URL for profiles
EXCEL_PATH = DATA_DIR / "mock_profiles.xlsx"
BASE_PROFILES_URL = "https://hamed-alikhani.github.io/mock-linkedin-profiles/profiles/"

###################
# DATA COLLECTION #
###################

def extract_profiles_from_excel():
    """Extract profile information from the Excel file"""
    if not EXCEL_PATH.exists():
        logger.error(f"Excel file not found at {EXCEL_PATH}")
        logger.info("Please download it first with: curl -L https://github.com/hamed-alikhani/mock-linkedin-profiles/raw/main/mock_profiles_aicompetition.xlsx -o data/mock_profiles.xlsx")
        return []
    
    try:
        # Read the Excel file
        logger.info(f"Reading Excel file from {EXCEL_PATH}")
        df = pd.read_excel(EXCEL_PATH)
        
        # Extract profile information
        profiles = []
        for _, row in df.iterrows():
            profile = {
                "name": row.get("Name", ""),
                "job_title": row.get("Job Title", ""),
                "contact_number": row.get("Contact Number", ""),
                "available_spot": row.get("Available Spot", ""),
                "city": row.get("City", ""),
                "linkedin_url": row.get("LinkedIn Profile", "")  # Fixed column name
            }
            
            # Extract the profile HTML filename from the LinkedIn URL
            if profile["linkedin_url"]:
                # Handle different URL formats
                if "/profiles/" in profile["linkedin_url"]:
                    profile["profile_filename"] = profile["linkedin_url"].split("/profiles/")[-1]
                else:
                    # Create a filename based on the name if URL doesn't contain the expected pattern
                    name_for_file = profile["name"].lower().replace(" ", "_")
                    profile["profile_filename"] = f"{name_for_file}.html"
            
            profiles.append(profile)
        
        logger.info(f"Extracted {len(profiles)} profiles from Excel file")
        return profiles
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        return []

def download_profile(profile):
    """Download an individual profile page and save as HTML"""
    if not profile.get("profile_filename"):
        logger.warning(f"Missing profile filename for {profile.get('name', 'Unknown')}")
        return None
    
    profile_url = f"{BASE_PROFILES_URL}{profile['profile_filename']}"
    filename = PROFILES_DIR / profile["profile_filename"]
    
    logger.info(f"Downloading profile: {profile_url}")
    
    try:
        response = requests.get(profile_url, timeout=10)
        response.raise_for_status()
        
        with open(filename, "w", encoding="utf-8") as file:
            file.write(response.text)
        
        logger.info(f"Saved profile to {filename}")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {profile_url}: {e}")
        return None

def extract_profile_data(html_content, profile_info):
    """Extract detailed information from the profile HTML and merge with Excel data"""
    if not html_content:
        return profile_info
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Add base info from Excel
    profile_data = {
        "name": profile_info.get("name", "Unknown"),
        "headline": profile_info.get("job_title", ""),
        "contact_number": profile_info.get("contact_number", ""),
        "available_spot": profile_info.get("available_spot", ""),
        "city": profile_info.get("city", ""),
        "source_url": profile_info.get("linkedin_url", ""),
    }
    
    # Extract about section
    try:
        about = soup.select_one('div.display-flex.ph5.pv3 div.pv-shared-text-with-see-more div.inline-show-more-text')
        if about:
            profile_data["about"] = about.text.strip()
    except (AttributeError, TypeError):
        profile_data["about"] = ""
    
    # Extract experience
    experience = []
    try:
        exp_sections = soup.select('div.pvs-list__outer-container ul.pvs-list li.artdeco-list__item')
        for exp in exp_sections:
            job_title_elem = exp.select_one('span.mr1.t-bold')
            company_elem = exp.select_one('span.t-14.t-normal')
            duration_elem = exp.select_one('span.t-14.t-normal.t-black--light')
            
            if job_title_elem and company_elem:
                job_title = job_title_elem.text.strip()
                company = company_elem.text.strip()
                duration = duration_elem.text.strip() if duration_elem else ""
                
                experience.append({
                    "title": job_title,
                    "company": company,
                    "duration": duration
                })
    except (AttributeError, TypeError):
        pass
    
    profile_data["experience"] = experience
    
    # Extract skills
    skills = []
    try:
        skill_sections = soup.select('div.pv-skill-categories-section ol.pv-skill-categories-section__top-skills > li')
        for skill in skill_sections:
            skill_name_elem = skill.select_one('p.pv-skill-category-entity__name')
            if skill_name_elem:
                skills.append(skill_name_elem.text.strip())
    except (AttributeError, TypeError):
        pass
    
    profile_data["skills"] = skills
    
    return profile_data

def scrape_profiles():
    """Main function to scrape all profiles"""
    # Extract profile information from Excel
    profile_list = extract_profiles_from_excel()
    
    if not profile_list:
        logger.warning("No profiles found in Excel file.")
        return
    
    logger.info(f"Preparing to scrape {len(profile_list)} profiles")
    
    # Download and process each profile
    processed_data = []
    
    for profile in profile_list:
        html_content = download_profile(profile)
        
        if html_content:
            # Extract detailed data and merge with Excel data
            profile_data = extract_profile_data(html_content, profile)
            
            if profile_data:
                processed_data.append(profile_data)
                
                # Save individual profile JSON
                json_filename = PROFILES_DIR / f"{profile['profile_filename'].replace('.html', '.json')}"
                with open(json_filename, "w", encoding="utf-8") as file:
                    json.dump(profile_data, file, indent=2)
                
                logger.info(f"Processed and saved data for {profile.get('name', 'Unknown')}")
            
            # Be nice to the server with a small delay
            time.sleep(1)
    
    # Save all profiles to a single JSON file
    all_profiles_file = DATA_DIR / "all_profiles.json"
    with open(all_profiles_file, "w", encoding="utf-8") as file:
        json.dump(processed_data, file, indent=2)
    
    logger.info(f"Completed scraping {len(processed_data)} profiles")
    logger.info(f"All profile data saved to {all_profiles_file}")

###################
# VECTOR DATABASE #
###################

def load_profile_data() -> List[Dict[Any, Any]]:
    """Load all profile data from the JSON file"""
    all_profiles_file = DATA_DIR / "all_profiles.json"
    
    if not all_profiles_file.exists():
        logger.error(f"Profile data file not found: {all_profiles_file}")
        return []
    
    with open(all_profiles_file, "r", encoding="utf-8") as file:
        profiles = json.load(file)
    
    logger.info(f"Loaded {len(profiles)} profiles from {all_profiles_file}")
    return profiles

def process_profile_to_text(profile: Dict[Any, Any]) -> str:
    """Convert a profile dictionary to a text format suitable for embedding"""
    texts = []
    
    # Basic information
    texts.append(f"Name: {profile.get('name', 'Unknown')}")
    texts.append(f"Title: {profile.get('headline', 'Unknown')}")
    
    # About section
    if profile.get('about'):
        texts.append(f"About: {profile['about']}")
    
    # Experience
    experiences = profile.get('experience', [])
    if experiences:
        texts.append("Experience:")
        for exp in experiences:
            exp_text = f"- {exp.get('title', '')} at {exp.get('company', '')}"
            if exp.get('duration'):
                exp_text += f" ({exp['duration']})"
            texts.append(exp_text)
    
    # Skills
    skills = profile.get('skills', [])
    if skills:
        texts.append("Skills: " + ", ".join(skills))
    
    # URL reference
    texts.append(f"Source: {profile.get('source_url', '')}")
    
    return "\n".join(texts)

def convert_profiles_to_documents(profiles: List[Dict[Any, Any]]) -> List[Document]:
    """Convert profile dictionaries to LangChain document objects with metadata"""
    documents = []
    
    for profile in profiles:
        profile_text = process_profile_to_text(profile)
        
        # Create a document with the profile text and metadata
        doc = Document(
            page_content=profile_text,
            metadata={
                "name": profile.get("name", "Unknown"),
                "title": profile.get("headline", "Unknown"),
                "source_url": profile.get("source_url", ""),
                "skills": ", ".join(profile.get("skills", [])),
                "profile_id": profile.get("name", "Unknown").lower().replace(" ", "_")
            }
        )
        documents.append(doc)
    
    logger.info(f"Converted {len(documents)} profiles to document objects")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False
    )
    
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs

def create_vector_db(api_key: str) -> Chroma:
    """Create a vector database from profile data"""
    # Ensure the API key is set
    if not api_key:
        logger.error("Google API key not provided")
        raise ValueError("Google API key is required")
    
    # Load profile data
    profiles = load_profile_data()
    
    if not profiles:
        logger.error("No profile data available to create vector database")
        raise ValueError("No profile data found")
    
    # Convert to documents
    documents = convert_profiles_to_documents(profiles)
    
    # Split documents (optional, depending on profile size)
    split_docs = split_documents(documents)
    
    # Create embeddings using Google's model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    
    # Create and persist the vector database
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    
    logger.info(f"Created vector database with {len(split_docs)} document chunks")
    logger.info(f"Vector database saved to {DB_DIR}")
    
    return vector_db

#################
# QUERY ENGINE  #
#################

def load_vector_db(api_key: str) -> Chroma:
    """Load the Chroma vector database"""
    logger.info(f"Loading vector database from {DB_DIR}")
    
    # Create embeddings using Google's model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    
    # Load the existing database
    vector_db = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    
    logger.success(f"Vector database loaded successfully")
    return vector_db

def create_prompt() -> ChatPromptTemplate:
    """Create the prompt template for candidate matching"""
    template = """
You are an AI Recruiter Assistant that matches job seekers with job requirements.

I'll provide you with a job requirement, and you'll need to analyze the profiles I have in my database
to find the best matches.

Job Requirement:
{query}

Here are the candidate profiles that might match:

{context}

Based on the above profiles and the job requirement, please:
1. Rank the top candidates (maximum 3) who best match the requirements.
2. For each candidate provide:
   - Name and current job title
   - Match score (1-100%)
   - Brief explanation of why they match the requirements
   - Key skills relevant to the position
   - Contact information

Overall summary: In 1-2 sentences, explain why these candidates are the best fit for the position.
"""

    return ChatPromptTemplate.from_template(template)

def create_query_engine(api_key: str):
    """Create the query engine for finding matching candidates"""
    # Load the vector database
    vector_db = load_vector_db(api_key)
    
    # Create a retriever from the vector database
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most similar documents
    )
    
    # Create the LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=api_key,
        temperature=0.2,  # Lower temperature for more consistent ranking
    )
    
    # Create the prompt template
    prompt = create_prompt()
    
    # Build the RAG pipeline
    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

def query_candidates(query: str, api_key: str) -> str:
    """Query the system for matching candidates based on job requirements"""
    logger.info(f"Querying for candidates matching: {query}")
    
    # Create the query engine
    engine = create_query_engine(api_key)
    
    # Run the query
    result = engine.invoke(query)
    
    return result

##################
# HELPER FUNCTIONS
##################

def check_profiles_data_exists() -> bool:
    """Check if profile data has been collected"""
    profiles_file = DATA_DIR / "all_profiles.json"
    return profiles_file.exists() and profiles_file.stat().st_size > 0

def check_vector_db_exists() -> bool:
    """Check if the vector database has been created"""
    # Check for the sqlite file and at least one embedding collection directory
    chroma_index = DB_DIR / "chroma.sqlite3"
    
    # Chroma creates a UUID directory for each collection
    collection_dirs = list(DB_DIR.glob("*-*-*-*-*"))
    
    # Check if the index file exists and at least one collection directory with binary files
    db_exists = (
        DB_DIR.exists() and 
        chroma_index.exists() and 
        len(collection_dirs) > 0 and
        any(len(list(d.glob("*.bin"))) > 0 for d in collection_dirs)
    )
    
    if db_exists:
        logger.info(f"Found existing Chroma database in {DB_DIR}")
    
    return db_exists

##################
# WEB INTERFACE
##################

def setup_streamlit():
    """Configure Streamlit"""
    # Configure logger to not interfere with Streamlit
    logger.remove()
    logger.add(lambda msg: None, level="INFO")  # Suppress log output to console
    
    # Page configuration
    st.set_page_config(
        page_title="AI Candidate Matcher",
        page_icon="🧑‍💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_environment():
    """Set up the environment, load API keys, and check for data and DB"""
    # Load environment variables
    load_dotenv()
    
    # Get the API keys from environment variables
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    # Initialize session state
    if "setup_done" not in st.session_state:
        st.session_state.setup_done = False
        st.session_state.api_key = api_key
        st.session_state.has_profiles = check_profiles_data_exists()
        st.session_state.has_vector_db = check_vector_db_exists()
    
    return st.session_state.api_key, st.session_state.has_profiles, st.session_state.has_vector_db

def initialize_data(api_key):
    """Initialize data and database if needed"""
    has_profiles = check_profiles_data_exists()
    has_vector_db = check_vector_db_exists()
    
    # If we don't have profiles, collect them
    if not has_profiles:
        with st.spinner("Collecting profile data..."):
            scrape_profiles()
        st.success("Profile data collected successfully!")
        has_profiles = True
    
    # If we don't have a vector database, create it
    if not has_vector_db and has_profiles:
        with st.spinner("Creating vector database..."):
            create_vector_db(api_key)
        st.success("Vector database created successfully!")
        has_vector_db = True
    
    return has_profiles and has_vector_db

def render_sidebar():
    """Render the sidebar content"""
    st.sidebar.title("AI Candidate Matcher")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application uses AI to match job requirements with candidate profiles.
    
    It uses:
    - RAG (Retrieval-Augmented Generation)
    - LangChain for the retrieval pipeline
    - Google's Generative AI for embeddings and inference
    - Chroma as the vector database
    
    ### Data Source
    The candidate data comes from mock LinkedIn profiles created for educational purposes.
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Developer Info")
    st.sidebar.info("Created with 💖 by AI-Austin-Agent")

def render_api_key_input():
    """Render the API key input form"""
    st.warning("⚠️ No Google API key found in environment!")
    
    with st.form("api_key_form"):
        api_key = st.text_input("Enter your Google API key", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted and api_key:
            st.session_state.api_key = api_key
            st.success("API key saved! Initializing system...")
            st.rerun()

def run_streamlit_app():
    """Main function for the Streamlit app"""
    # Import app.py to use its implementation instead
    try:
        import app
        app.main()
    except ImportError:
        # Fallback if app.py cannot be imported
        setup_streamlit()
        render_sidebar()
        
        # Setup environment and check for API key
        api_key, has_profiles, has_vector_db = setup_environment()
        
        # If no API key is found, show API key input form
        if not api_key:
            render_api_key_input()
            return
        
        # Initialize data if needed
        if not (has_profiles and has_vector_db):
            system_ready = initialize_data(api_key)
            if not system_ready:
                st.error("Failed to initialize the system. Please check the logs.")
                return
        
        # Show a message that the app should be run through app.py
        st.warning("This is a limited version of the app. For the full experience, please run 'streamlit run app.py'")
        st.info("This module provides the backend functionality only. The interface may be limited.")
        
        # Provide a basic search interface
        query = st.text_area("Job Requirements", height=150)
        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    result = query_candidates(query, api_key)
                st.write(result)

if __name__ == "__main__":
    run_streamlit_app() 