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
import re

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import Pydantic models
from models import Profile, Experience, CandidateMatch, SearchResponse

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
            # Get the raw HTML content of the experience item
            item_html = str(exp)
            
            # Extract title from the strong tag
            title_elem = exp.find('strong')
            title = title_elem.text.strip() if title_elem else ""
            
            # Extract company and location - it appears after the strong tag and before <br>
            company = ""
            company_loc_text = ""
            if title_elem and " at " in item_html:
                # Get the text after the strong tag closing and before <br>
                after_title = item_html.split('</strong>', 1)[1].split('<br>', 1)[0]
                if " at " in after_title:
                    company_loc_text = after_title.split(" at ", 1)[1].strip()
                    # Extract company (handle format like "Company, Location")
                    if "," in company_loc_text:
                        company = company_loc_text.split(",")[0].strip()
                    else:
                        company = company_loc_text
            
            # Extract duration
            duration_elem = exp.find('small')
            duration = duration_elem.text.strip() if duration_elem else ""
            
            # Extract description
            description_elem = exp.find('p')
            description = description_elem.text.strip() if description_elem else ""
            
            # Add to experience list
            experience.append({
                "title": title,
                "company": company,
                "duration": duration,
                "description": description
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

def extract_experience_and_skills():
    """
    Extract experience and skills data from HTML profiles and update JSON files
    Also stores raw HTML as fallback for more complex extraction needs
    """
    logger.info("Starting extraction of experience and skills from HTML profiles")
    
    # Ensure the profiles directory exists
    if not PROFILES_DIR.exists():
        logger.error(f"Profiles directory not found at {PROFILES_DIR}")
        return 0
    
    # Get all HTML files in the profiles directory
    html_files = list(PROFILES_DIR.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML profile files")
    
    # Load all profiles from JSON to update later
    all_profiles_file = DATA_DIR / "all_profiles.json"
    all_profiles = []
    if all_profiles_file.exists():
        try:
            with open(all_profiles_file, "r", encoding="utf-8") as file:
                all_profiles = json.load(file)
            logger.info(f"Loaded {len(all_profiles)} profiles from {all_profiles_file}")
        except Exception as e:
            logger.error(f"Error loading all profiles: {e}")
            all_profiles = []
    
    profiles_updated = 0
    
    # Process each HTML file
    for html_file in html_files:
        profile_name = html_file.stem  # Get filename without extension
        json_file = PROFILES_DIR / f"{profile_name}.json"
        
        # Skip if JSON file doesn't exist
        if not json_file.exists():
            logger.warning(f"JSON file not found for {profile_name}, skipping")
            continue
        
        try:
            # Read HTML content
            with open(html_file, "r", encoding="utf-8") as file:
                html_content = file.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract experience - find all sections and identify the Experience section
            experience = []
            sections = soup.find_all('div', class_='section')
            for section in sections:
                # Check if this section is the Experience section
                h2_elem = section.find('h2')
                if h2_elem and 'Experience' in h2_elem.text:
                    # Found the Experience section
                    exp_items = section.find_all('div', class_='experience-item')
                    for item in exp_items:
                        # Get the raw HTML content of the experience item
                        item_html = str(item)
                        
                        # Extract title from the strong tag
                        title_elem = item.find('strong')
                        title = title_elem.text.strip() if title_elem else ""
                        
                        # Extract company and location - it appears after the strong tag and before <br>
                        company = ""
                        company_loc_text = ""
                        if title_elem and " at " in item_html:
                            # Get the text after the strong tag closing and before <br>
                            after_title = item_html.split('</strong>', 1)[1].split('<br>', 1)[0]
                            if " at " in after_title:
                                company_loc_text = after_title.split(" at ", 1)[1].strip()
                                # Extract company (handle format like "Company, Location")
                                if "," in company_loc_text:
                                    company = company_loc_text.split(",")[0].strip()
                                else:
                                    company = company_loc_text
                        
                        # Extract duration
                        duration_elem = item.find('small')
                        duration = duration_elem.text.strip() if duration_elem else ""
                        
                        # Extract description
                        description_elem = item.find('p')
                        description = description_elem.text.strip() if description_elem else ""
                        
                        # Add to experience list
                        experience.append({
                            "title": title,
                            "company": company,
                            "duration": duration,
                            "description": description
                        })
                    # Found and processed Experience section, no need to continue
                    break
            
            # Extract skills - find all sections and identify the Skills section
            skills = []
            for section in sections:
                # Check if this section is the Skills section
                h2_elem = section.find('h2')
                if h2_elem and 'Skills' in h2_elem.text:
                    # Found the Skills section
                    skills_list = section.find('ul')
                    if skills_list:
                        for skill_item in skills_list.find_all('li'):
                            skill_text = skill_item.text.strip()
                            # Extract skill name (remove endorsements count)
                            skill_name = skill_text
                            if "(" in skill_text:
                                skill_name = skill_text.split("(")[0].strip()
                            skills.append(skill_name)
                    # Found and processed Skills section, no need to continue
                    break
            
            # Update individual JSON file
            profile_data = {}
            try:
                with open(json_file, "r", encoding="utf-8") as file:
                    profile_data = json.load(file)
            except Exception as e:
                logger.error(f"Error reading JSON file {json_file}: {e}")
                continue
            
            # Update experience and skills
            profile_data["experience"] = experience
            profile_data["skills"] = skills
            
            # Store raw HTML as fallback for more complex extraction needs
            profile_data["raw_html"] = html_content
            
            # Save updated JSON
            with open(json_file, "w", encoding="utf-8") as file:
                json.dump(profile_data, file, indent=2)
            
            # Update in all_profiles
            for i, profile in enumerate(all_profiles):
                if profile.get("name") == profile_data.get("name"):
                    all_profiles[i]["experience"] = experience
                    all_profiles[i]["skills"] = skills
                    all_profiles[i]["raw_html"] = html_content
                    break
            
            profiles_updated += 1
            logger.info(f"Updated profile for {profile_name}")
            
        except Exception as e:
            logger.error(f"Error processing {profile_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Save updated all_profiles.json
    try:
        with open(all_profiles_file, "w", encoding="utf-8") as file:
            json.dump(all_profiles, file, indent=2)
        logger.info(f"Updated all_profiles.json with experience and skills data")
    except Exception as e:
        logger.error(f"Error saving all_profiles.json: {e}")
    
    logger.info(f"Completed! Updated {profiles_updated} profiles out of {len(html_files)} HTML files")
    return profiles_updated

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
    texts.append(f"Contact: {profile.get('contact_number', '')}")
    texts.append(f"Available: {profile.get('available_spot', '')}")
    texts.append(f"Location: {profile.get('city', '')}")
    
    # About section
    if profile.get('about'):
        texts.append(f"About: {profile['about']}")
    
    # Experience in more detail
    experiences = profile.get('experience', [])
    if experiences:
        texts.append("Experience:")
        for exp in experiences:
            exp_parts = []
            exp_parts.append(f"- {exp.get('title', '')}")
            if exp.get('company'):
                exp_parts.append(f"at {exp.get('company', '')}")
            if exp.get('duration'):
                exp_parts.append(f"({exp['duration']})")
            
            texts.append(" ".join(exp_parts))
            
            # Include the description for richer context
            if exp.get('description'):
                texts.append(f"  Description: {exp['description']}")
    
    # Skills with more details
    skills = profile.get('skills', [])
    if skills:
        texts.append("Skills: " + ", ".join(skills))
    
    # URL reference
    texts.append(f"Source: {profile.get('source_url', '')}")
    
    # Note about raw HTML availability (for LLM awareness)
    if profile.get('raw_html'):
        texts.append("Note: Raw HTML profile data is available for more detailed information extraction.")
    
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
2. DO NOT include any overall summary at the beginning of your response.
3. For each candidate provide:
   - Name and current job title
   - Match score (1-100%)
   - Brief explanation of why they match the requirements
   - Key skills relevant to the position (from both their skills list and experience)
   - Contact information (phone number from their profile)
4. ONLY at the end of your response, provide a brief overall summary explaining why these candidates are the best fit.

IMPORTANT: The profiles provided include full details about each candidate, including:
- Their job experience (company, title, duration, and description)
- Their listed skills
- Contact information

For some profiles, there may also be raw HTML data stored in the "raw_html" field.
Use all available information to make the best match.

CRITICAL INSTRUCTION: YOU MUST FORMAT YOUR RESPONSE AS JSON. Do not include any text outside the JSON structure.

The exact JSON format to use is:
```json
{{
  "candidates": [
    {{
      "profile": {{
        "name": "Candidate Name",
        "headline": "Job Title",
        "contact_number": "Phone Number",
        "available_spot": "Available Time",
        "city": "City, State",
        "source_url": "Profile URL"
      }},
      "match_score": 95,
      "explanation": "Reason why this candidate matches the requirements",
      "relevant_skills": ["Skill 1", "Skill 2", "Skill 3"]
    }}
  ],
  "summary": "Brief summary explaining why these candidates are the best fit"
}}
```

Do not include any markdown formatting or explanatory text outside the JSON. Your entire response should be valid JSON that can be parsed directly.
"""

    return ChatPromptTemplate.from_template(template)

def parse_unstructured_response(text: str) -> Dict[str, Any]:
    """
    Parse an unstructured response into a dictionary that can be used to create a SearchResponse.
    This is a fallback when the response is not valid JSON.
    """
    candidates = []
    summary = ""
    
    # Clean up the text
    # More aggressively remove any summary section at the beginning
    cleaned_text = re.sub(r'^.*?(?:Summary|Overall\s+summary)[:\s].*?\n\n', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Extract summary - look ONLY for the section at the very end
    # Look specifically for "Overall summary:" or "Summary:" at the end of the document
    summary_pattern = r'\n(?:Overall\s+summary|Summary)[:\s]+(.*?)$'
    summary_match = re.search(summary_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()
        
        # Remove the extracted summary from the text to avoid confusion in candidate extraction
        cleaned_text = cleaned_text[:summary_match.start()]
    
    # Extract candidates
    # This regex looks for numbered candidates with name and title
    candidate_pattern = r'(?:^|\n)#?(\d+)[\.:\)]\s+([^(]+)(?:\(([^)]+)\))?'
    candidate_matches = re.finditer(candidate_pattern, cleaned_text)
    
    for match in candidate_matches:
        rank = match.group(1)
        name = match.group(2).strip() if match.group(2) else "Unknown"
        title = match.group(3).strip() if match.group(3) else ""
        
        # Get the candidate text section - from this match to next candidate or end
        start_pos = match.end()
        next_match = re.search(candidate_pattern, cleaned_text[start_pos:])
        end_pos = (next_match.start() + start_pos) if next_match else len(cleaned_text)
        candidate_text = cleaned_text[start_pos:end_pos]
        
        # Extract match score
        score_pattern = r'(?:Match\s+score|Score):\s*(\d+)%'
        score_match = re.search(score_pattern, candidate_text)
        score = int(score_match.group(1)) if score_match and score_match.group(1).isdigit() else 0
        
        # Extract explanation
        explanation_pattern = r'(?:Explanation|Why|Reason)(?:\s+this\s+candidate\s+matches)?:\s*(.+?)(?:\n\n|\n(?:Key skills|Skills|Contact)|$)'
        explanation_match = re.search(explanation_pattern, candidate_text, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        
        # Extract skills
        skills_pattern = r'(?:Key\s+skills|Skills):\s*(.+?)(?:\n\n|\n(?:Contact|Match|#\d+|$))'
        skills_match = re.search(skills_pattern, candidate_text, re.DOTALL | re.IGNORECASE)
        skills_text = skills_match.group(1).strip() if skills_match else ""
        skills = [s.strip() for s in re.split(r',|\n|\*|\-', skills_text) if s.strip()]
        
        # Extract contact
        contact_pattern = r'Contact(?:\s+information)?:\s*(.+?)(?:\n\n|\n(?:#\d+|\d\.|\Z))'
        contact_match = re.search(contact_pattern, candidate_text, re.DOTALL | re.IGNORECASE)
        contact = contact_match.group(1).strip() if contact_match else ""
        
        # Create candidate entry
        candidates.append({
            "profile": {
                "name": name,
                "headline": title,
                "contact_number": contact,
                "available_spot": "",
                "city": "",
                "source_url": ""
            },
            "match_score": score if score > 0 else 50,  # Default to 50% if no score found
            "explanation": explanation,
            "relevant_skills": skills
        })
    
    return {
        "candidates": candidates,
        "summary": summary
    }

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
    
    # Try to parse as JSON first
    try:
        # Extract JSON portion from the result - the LLM may include text before/after
        # Find first { and last }
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_result = result[json_start:json_end]
            try:
                parsed_result = json.loads(json_result)
                logger.info("Successfully parsed JSON response")
            except json.JSONDecodeError:
                # Try cleaning up the JSON before parsing
                import re
                # Replace any JavaScript comments
                cleaned_json = re.sub(r'//.*?\n', '', json_result)
                # Try parsing again
                try:
                    parsed_result = json.loads(cleaned_json)
                    logger.info("Successfully parsed cleaned JSON response")
                except json.JSONDecodeError:
                    # If still fails, fall back to unstructured parsing
                    logger.warning("Failed to parse JSON, falling back to unstructured parsing")
                    parsed_result = parse_unstructured_response(result)
        else:
            # If no JSON-like structure found, use unstructured parsing
            logger.warning("No JSON structure found, using unstructured parsing")
            parsed_result = parse_unstructured_response(result)
            
        # Create a structured response using Pydantic models
        candidates = []
        for candidate in parsed_result.get("candidates", []):
            # Create Profile instance 
            profile_data = candidate.get("profile", {})
            
            # Create Profile with available data
            try:
                profile = Profile(
                    name=profile_data.get("name", ""),
                    headline=profile_data.get("headline", ""),
                    contact_number=profile_data.get("contact_number") or "",
                    available_spot=profile_data.get("available_spot") or "",
                    city=profile_data.get("city") or "",
                    source_url=profile_data.get("source_url") or "",
                    experience=profile_data.get("experience", []),  # Let the model validator handle this
                    skills=profile_data.get("skills", []),
                    about=profile_data.get("about", "")
                )
                
                # Create CandidateMatch
                candidates.append(CandidateMatch(
                    profile=profile,
                    match_score=candidate.get("match_score", 0),
                    explanation=candidate.get("explanation", ""),
                    relevant_skills=candidate.get("relevant_skills", [])
                ))
            except Exception as e:
                logger.error(f"Error creating profile from data: {e}")
                logger.debug(f"Profile data: {profile_data}")
        
        # Create SearchResponse
        search_response = SearchResponse(
            candidates=candidates,
            summary=parsed_result.get("summary", "")
        )
        
        # Return a formatted string representation
        return search_response.to_formatted_response()
        
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        # If all parsing fails, return the original result
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