import os
import json
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
import re
import pandas as pd
import time
import html

# Import our modules
import ai_agent

# Configure logger to not interfere with Streamlit
logger.remove()
logger.add(lambda msg: None, level="INFO")  # Suppress log output to console

# Define constants
DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
PROFILES_FILE = DATA_DIR / "all_profiles.json"

# Page configuration
st.set_page_config(
    page_title="AI Candidate Matcher Pro",
    page_icon="🧑‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_css():
    """Load custom CSS"""
    st.markdown("""
    <style>
    .candidate-card {
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .candidate-name {
        color: #0366d6;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .candidate-title {
        color: #586069;
        font-size: 16px;
        margin-bottom: 15px;
    }
    .candidate-score {
        color: #28a745;
        font-weight: bold;
    }
    .candidate-section {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .section-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .skill-tag {
        background-color: #e1e4e8;
        color: #24292e;
        border-radius: 15px;
        padding: 5px 10px;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
        font-size: 0.85em;
    }
    </style>
    """, unsafe_allow_html=True)

def load_profiles_as_df():
    """Load profiles from JSON and convert to DataFrame for easy filtering"""
    try:
        if PROFILES_FILE.exists():
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                profiles = json.load(f)
                
            # Create a DataFrame
            df_profiles = pd.DataFrame(profiles)
            
            # Extract skills as a separate column
            df_profiles["skills_count"] = df_profiles["skills"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            df_profiles["skills_text"] = df_profiles["skills"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
            
            # Extract experience count
            df_profiles["experience_count"] = df_profiles["experience"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            
            return df_profiles
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading profiles: {e}")
        return None

def parse_candidate_results(result_text):
    """Parse the AI-generated results into structured data for better display"""
    # Initialize empty list for candidates
    candidates = []
    
    try:
        # Extract candidate sections using regex pattern for numbered items
        # This pattern looks for lines starting with a number followed by a period (1., 2., etc.)
        name_pattern = r'(?:^|\n)(\d+)[\.:\)]\s+([^(]+)(?:\(([^)]+)\))?'
        
        # Find all candidates first to know the total count
        all_matches = list(re.finditer(name_pattern, result_text))
        total_matches = len(all_matches)
        
        # Process each candidate match
        for index, match in enumerate(all_matches):
            # Extract basic info
            rank = match.group(1)
            name = match.group(2).strip() if match.group(2) else "Unknown"
            title = match.group(3).strip() if match.group(3) else ""
            
            # Determine the section text for this candidate
            # (from the end of this match to the start of the next match or end of text)
            start_pos = match.end()
            
            # Find the next candidate or the end of the text
            if index < total_matches - 1:
                # There is another candidate after this one
                next_match = all_matches[index + 1]
                end_pos = next_match.start()
            else:
                # This is the last candidate
                end_pos = len(result_text)
            
            # Extract the candidate section text
            section_text = result_text[start_pos:end_pos]
            
            # Extract match score - look for any number followed by %
            score_pattern = r'(?:Match score|Score):\s*(\d+)%'
            score_match = re.search(score_pattern, section_text)
            score = score_match.group(1) if score_match else "N/A"
            
            # Extract explanation
            explanation_pattern = r'(?:Explanation|Why|Reason)(?:\s+this\s+candidate\s+matches)?:\s*(.+?)(?=\n\s*(?:Key skill|Skill|Contact|Match score)|$)'
            explanation_match = re.search(explanation_pattern, section_text, re.IGNORECASE | re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            explanation = re.sub(r'\n+', ' ', explanation)  # Clean up newlines
            
            # Extract skills
            skills_pattern = r'(?:Key skill|Skill|Relevant skill)s?:\s*(.+?)(?=\n\s*(?:Contact|Match score|Explanation|Why)|$)'
            skills_match = re.search(skills_pattern, section_text, re.IGNORECASE | re.DOTALL)
            skills_text = skills_match.group(1).strip() if skills_match else ""
            skills_text = re.sub(r'\n+', ' ', skills_text)  # Clean up newlines
            # Split skills by commas, bullets, or similar delimiters
            skills_list = [s.strip() for s in re.split(r',|\n|•|-', skills_text) if s.strip()]
            
            # Make sure we don't capture SUMMARY text in the contact information
            contact_pattern = r'Contact(?:\s+information)?:\s*(.*?)(?=\n\s*(?:Key skill|Skill|Match score|Explanation|Why|Summary|SUMMARY:)|\n\n|\Z)'
            contact_match = re.search(contact_pattern, section_text, re.IGNORECASE | re.DOTALL)
            contact = contact_match.group(1).strip() if contact_match else "No contact information available"
            contact = re.sub(r'\n+', ' ', contact)  # Clean up newlines
            
            # Clean up contact info - remove any SUMMARY text that might have been captured
            contact = re.sub(r'(?i)SUMMARY:.*$', '', contact).strip()
            
            # Add to candidates list
            candidates.append({
                "rank": rank,
                "name": name,
                "title": title,
                "score": score,
                "explanation": explanation,
                "skills": skills_list,
                "contact": contact
            })
    
    except Exception as e:
        st.error(f"Error parsing candidates: {e}")
        return [], ""
    
    # Extract summary if present
    summary = ""
    try:
        # Look for various summary formats, including the new SUMMARY format
        summary_pattern = r'(?:Overall summary|In summary|Summary|SUMMARY):\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\Z)'
        summary_match = re.search(summary_pattern, result_text, re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
            summary = re.sub(r'\n+', ' ', summary)
    except Exception:
        pass
    
    return candidates, summary

def render_candidate_card(candidate):
    """Render a candidate as a card with proper styling"""
    # Create a cleaner display for candidates without using HTML directly
    st.markdown(f"### #{candidate['rank']} {candidate['name']}")
    st.markdown(f"**{candidate['title']}**")
    st.markdown(f"**Match score:** {candidate['score']}%")
    
    st.markdown("**Why this candidate matches:**")
    st.write(candidate["explanation"])
    
    st.markdown("**Key skills:**")
    # Display skills as tags using Streamlit's built-in columns
    if candidate["skills"]:
        cols = st.columns(4)
        for i, skill in enumerate(candidate["skills"]):
            with cols[i % 4]:
                st.markdown(f"<span class='skill-tag'>{html.escape(skill)}</span>", unsafe_allow_html=True)
    else:
        st.write("No skills listed")
    
    st.markdown("**Contact:**")
    st.write(candidate["contact"])
    
    # Add a visual separator
    st.markdown("---")

def display_database_stats():
    """Display statistics about the database"""
    # Load profiles for stats
    df_profiles = load_profiles_as_df()
    
    if df_profiles is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Profiles", len(df_profiles))
        
        with col2:
            avg_skills = df_profiles["skills_count"].mean()
            st.metric("Avg Skills per Profile", f"{avg_skills:.1f}")
        
        with col3:
            avg_exp = df_profiles["experience_count"].mean()
            st.metric("Avg Experience per Profile", f"{avg_exp:.1f}")
        
        # Top skills bar chart
        if not df_profiles.empty and "skills" in df_profiles.columns:
            # Flatten the skills lists and count occurrences
            all_skills = []
            for skills in df_profiles["skills"]:
                if isinstance(skills, list):
                    all_skills.extend(skills)
            
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts().head(10)
                
                st.subheader("Top 10 Skills in Database")
                st.bar_chart(skill_counts)

def render_sidebar():
    """Render the sidebar content"""
    st.sidebar.title("AI Candidate Matcher Pro")
    st.sidebar.markdown("---")
    
    # Show database status
    has_profiles = ai_agent.check_profiles_data_exists()
    has_vector_db = ai_agent.check_vector_db_exists()
    
    status_col1, status_col2 = st.sidebar.columns(2)
    
    with status_col1:
        st.markdown("**Profile Data:**")
        if has_profiles:
            st.success("✅ Available")
        else:
            st.error("❌ Missing")
    
    with status_col2:
        st.markdown("**Vector Database:**")
        if has_vector_db:
            st.success("✅ Available")
        else:
            st.error("❌ Missing")
    
    # Actions section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Actions")
    
    if st.sidebar.button("Refresh Data"):
        with st.sidebar:
            with st.spinner("Checking data status..."):
                time.sleep(1)  # Small delay for UI feedback
                st.session_state.has_profiles = ai_agent.check_profiles_data_exists()
                st.session_state.has_vector_db = ai_agent.check_vector_db_exists()
                st.rerun()
    
    # About section
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
    
    # Debug mode toggle
    st.sidebar.markdown("---")
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    
    st.sidebar.checkbox("Debug Mode", key="debug_mode")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Developer Info")
    st.sidebar.info("Created with 💖 by AI-Austin-Agent")

def render_api_key_input():
    """Render the API key input form"""
    st.warning("⚠️ No Google API key found in environment!")
    
    st.markdown("""
    To use this application, you need a Google API key with access to the Gemini API.
    You can get one by:
    1. Going to [Google AI Studio](https://ai.google.dev/)
    2. Creating an account or signing in
    3. Navigating to the API Keys section
    4. Creating a new API key
    """)
    
    with st.form("api_key_form"):
        api_key = st.text_input("Enter your Google API key", type="password", 
                              help="Your API key will be stored only for this session and not saved permanently")
        submitted = st.form_submit_button("Submit")
        
        if submitted and api_key:
            st.session_state.api_key = api_key
            st.success("API key saved! Initializing system...")
            st.rerun()

def render_candidate_search(api_key):
    """Render the candidate search interface"""
    st.title("🧑‍💻 AI Candidate Matcher Pro")
    
    # Initialize session state for tracking search state and results
    if "active_search" not in st.session_state:
        st.session_state.active_search = False
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "from_example" not in st.session_state:
        st.session_state.from_example = False
    
    # Show database statistics in an expander
    with st.expander("📊 Database Statistics", expanded=False):
        display_database_stats()
    
    st.markdown("""
    Enter job requirements and find the best matching candidates from our database.
    The AI will analyze the requirements and rank the most suitable candidates.
    """)
    
    # Job requirements input
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    # Helper functions for example queries
    def set_example_query(example_text):
        st.session_state.query = example_text
        st.session_state.from_example = True
        st.rerun()
    
    query = st.text_area(
        "Job Requirements", 
        value=st.session_state.query,
        height=150,
        placeholder="Example: Senior Data Engineer with 5+ years experience in ETL pipeline development, Python, and cloud technologies."
    )
    
    # Add example queries
    example_container = st.container()
    example_container.markdown("**Example queries:**")
    col1, col2, col3 = example_container.columns(3)
    
    # Example query buttons
    if col1.button("Data Engineer with ETL experience"):
        set_example_query("I need a Senior Data Engineer with experience in ETL pipelines and data warehousing.")
    
    if col2.button("Machine Learning Engineer"):
        set_example_query("Looking for a Machine Learning Engineer with Python experience and knowledge of neural networks.")
    
    if col3.button("Frontend Developer"):
        set_example_query("Need a Frontend Developer with React, JavaScript, and UI/UX skills.")
    
    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        num_results = st.slider("Number of candidates to retrieve", min_value=1, max_value=10, value=3)
        enhance_formatting = st.checkbox("Enhance AI response formatting", value=True, 
                                       help="Add additional instructions to improve the structure of AI responses")
    
    # Search button
    search_clicked = st.button("🔍 Find Matching Candidates", type="primary")
    
    # Auto-trigger search if query came from example
    auto_search = st.session_state.from_example
    if auto_search:
        st.session_state.from_example = False  # Reset flag
    
    # Determine if we should search
    should_search = search_clicked or st.session_state.active_search or auto_search
    
    if should_search and query:
        # Reset the active search flag
        st.session_state.active_search = False
        
        with st.spinner("Searching for matching candidates..."):
            # Inject the number of candidates and formatting instructions into the query
            augmented_query = f"{query}\n\nPlease return the top {num_results} candidates."
            
            if enhance_formatting:
                augmented_query += """
                
Please format your response exactly as follows:

1. John Smith (Software Engineer)
Match score: 85%
Explanation: John has 8 years of experience in software engineering with a focus on backend development. He has worked extensively with cloud technologies and has led multiple successful projects.
Key skills: Python, Java, Cloud Architecture, AWS, System Design
Contact information: john.smith@example.com, (555) 123-4567

2. Jane Doe (Full Stack Developer)
Match score: 78%
Explanation: Jane has strong full-stack development skills with 5 years of experience in building web applications. She has demonstrated expertise in both frontend and backend technologies.
Key skills: JavaScript, React, Node.js, MongoDB, REST APIs
Contact information: jane.doe@example.com, (555) 987-6543

Overall summary: These candidates offer a strong mix of technical skills and experience that align well with the job requirements.

DO NOT include any HTML tags, markdown formatting, or additional text. Stick strictly to this format.
"""
            
            result = ai_agent.query_candidates(augmented_query, api_key)
            
            # Store the result for display
            st.session_state.current_result = result
        
        # Clean any HTML tags from the result that might cause rendering issues
        cleaned_result = re.sub(r'<[^>]*>', '', st.session_state.current_result)
        
        # Parse and display results
        candidates, summary = parse_candidate_results(cleaned_result)
        
        # Remove the SUMMARY section from the end of the result before displaying
        # This prevents the summary from appearing after the last contact
        cleaned_for_display = re.sub(r'\nSUMMARY:.*$', '', cleaned_result, flags=re.IGNORECASE | re.DOTALL)
        
        # If we got no candidates from the original parse, try parsing the cleaned version
        if not candidates:
            candidates, _ = parse_candidate_results(cleaned_for_display)
            
        st.markdown("### 🏆 Matching Candidates")
        
        if candidates:
            # Show summary first if available
            if summary:
                st.markdown("#### Summary")
                st.info(summary)
            else:
                # Debug information if no summary was found
                if st.session_state.debug_mode:
                    st.warning("No summary was extracted from the response.")
            
            # Display each candidate as a card
            for candidate in candidates:
                render_candidate_card(candidate)
            
            # Show raw results in debug mode
            if st.session_state.debug_mode:
                with st.expander("Raw AI Response"):
                    st.code(cleaned_result, language="text")
                with st.expander("Cleaned AI Response (without summary)"):
                    st.code(cleaned_for_display, language="text")
        else:
            # Fall back to displaying raw results if parsing fails
            st.warning("The AI response couldn't be parsed into the structured format. Displaying raw results instead.")
            # Use write instead of markdown for safer rendering of plain text
            st.write(cleaned_for_display)
            if st.session_state.debug_mode:
                with st.expander("Raw AI Response"):
                    st.code(cleaned_result, language="text")

def initialize_data(api_key):
    """Initialize data and database if needed"""
    has_profiles = ai_agent.check_profiles_data_exists()
    has_vector_db = ai_agent.check_vector_db_exists()
    
    # If we don't have profiles, collect them
    if not has_profiles:
        with st.spinner("Collecting profile data... This may take a few minutes."):
            ai_agent.scrape_profiles()
        st.success("Profile data collected successfully!")
        has_profiles = True
    
    # If we don't have a vector database, create it
    if not has_vector_db and has_profiles:
        with st.spinner("Creating vector database... This may take a few minutes."):
            ai_agent.create_vector_db(api_key)
        st.success("Vector database created successfully!")
        has_vector_db = True
    
    return has_profiles and has_vector_db

def main():
    """Main function for the Streamlit app"""
    # Load custom CSS
    load_css()
    
    # Add additional CSS fixes for skill tags
    st.markdown("""
    <style>
    /* Additional fixes for skill tags */
    .skill-tag {
        background-color: #e1e4e8;
        color: #24292e;
        border-radius: 15px;
        padding: 5px 10px;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
        font-size: 0.85em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Setup environment and check for API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    # Initialize session state
    if "setup_done" not in st.session_state:
        st.session_state.setup_done = False
        st.session_state.api_key = api_key
        st.session_state.has_profiles = ai_agent.check_profiles_data_exists()
        st.session_state.has_vector_db = ai_agent.check_vector_db_exists()
    
    # Use API key from session state if available
    api_key = st.session_state.api_key
    
    # If no API key is found, show API key input form
    if not api_key:
        render_api_key_input()
        return
    
    # Initialize data if needed
    has_profiles = st.session_state.has_profiles
    has_vector_db = st.session_state.has_vector_db
    
    if not (has_profiles and has_vector_db):
        system_ready = initialize_data(api_key)
        if not system_ready:
            st.error("Failed to initialize the system. Please check the logs.")
            return
        
        # Update session state
        st.session_state.has_profiles = True
        st.session_state.has_vector_db = True
    
    # Render the candidate search interface
    render_candidate_search(api_key)

if __name__ == "__main__":
    main() 