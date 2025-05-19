# AI Candidate Matcher

An AI-powered application that matches candidates from LinkedIn profiles to job requirements using a RAG (Retrieval-Augmented Generation) pipeline with Google's Gemini AI model.

## Features

- **Semantic Search**: Match job requirements to candidate profiles using state-of-the-art AI
- **Visual Result Cards**: View candidate matches in beautifully formatted cards
- **Skills Tagging**: Easily identify key skills with visual tags
- **Database Statistics**: Visualize your candidate database with interactive charts
- **Customizable Search**: Control the number of results and search parameters
- **Debug Mode**: Investigate raw AI responses for troubleshooting
- **Mobile-Friendly UI**: Responsive design that works on all devices

## Project Overview

This application uses RAG (Retrieval-Augmented Generation) to:
1. Process mock LinkedIn profiles
2. Store profile information in a vector database (Chroma)
3. Match the most relevant candidates to job requirements input by the user

## Technology Stack

- **LangChain**: For building the RAG pipeline
- **Chroma Vector Database**: For storing embeddings of candidate profiles
- **Google Gemini AI**: For generating embeddings and inference
- **Streamlit**: For the web interface
- **BeautifulSoup**: For HTML parsing and data extraction
- **Pandas**: For data processing and visualization

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/ai-austin-agent.git
cd ai-austin-agent
```

2. Install dependencies:

```bash
pip install -e .
```

3. Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important:** This file is listed in `.gitignore` and should never be committed to version control to keep your API key secure.

### API Key Management

If working in a team, it's recommended to:

1. Use the provided `.env-template` file as a template
2. Copy it to create your own `.env` file:
   ```bash
   cp .env-template .env
   ```
3. Add your personal API key to the `.env` file

#### Alternative: Using Environment Variables

If you prefer not to use a .env file, you can set the environment variable directly in your shell:

```bash
# For Linux/Mac
export GEMINI_API_KEY=your_gemini_api_key_here

# For Windows (Command Prompt)
set GEMINI_API_KEY=your_gemini_api_key_here

# For Windows (PowerShell)
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

## Running the Application

### Web Interface

Run the web application with the provided script:

```bash
./run.sh
```

Or manually start the Streamlit server:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your browser.

### Command Line Interface

Alternatively, use the command-line interface:

```bash
python main.py
```

The CLI automatically:
1. Checks if profile data exists, and collects it if needed
2. Checks if the vector database exists, and creates it if needed

### Testing the Model

You can test your connection to the Gemini model:

```bash
python main.py --test-model
```

## Usage Guide

1. **Enter Job Requirements**: Type or paste the job requirements in the text box
2. **Customize Search**: Expand the "Advanced Options" section to adjust parameters
3. **View Database Stats**: Check the database statistics in the expandable section
4. **Get Matches**: Click "Find Matching Candidates" to see AI-matched results
5. **Analyze Results**: Review the ranked candidates with match explanations
6. **Debug If Needed**: Toggle debug mode in the sidebar to see raw AI responses

## Project Structure

```
ai-austin-agent/
├── data/                   # Directory for storing collected profile data
│   ├── profiles/           # Individual profile HTML and JSON files
│   └── all_profiles.json   # Consolidated JSON file with all profiles
├── chroma_db/              # Vector database storage
├── ai_agent.py             # Consolidated implementation of all functionality
├── app.py                  # Streamlit web interface
├── main.py                 # Command-line interface
├── pyproject.toml          # Project dependencies
├── README.md               # Project documentation
├── .env-template           # Template for creating .env file with API keys
├── run.sh                  # Shell script to run the Streamlit app
├── .gitignore              # Git ignore configuration
└── .python-version         # Python version specification
```

## Implementation Details

### Core Components

- **`ai_agent.py`**: Single-file implementation containing:
  - Data collection (scraping LinkedIn profiles)
  - Vector database management (Chroma)
  - RAG query pipeline (LangChain)
  - Helper functions for environment and API setup

- **`app.py`**: Enhanced Streamlit web interface with:
  - Visual candidate cards with skill tags
  - Database statistics visualization
  - Advanced search options
  - Debug mode for troubleshooting

- **`main.py`**: Command-line interface supporting:
  - Interactive query mode
  - Direct query processing
  - Model testing functionality
  - Automatic data initialization

- **`run.sh`**: Helper script to:
  - Check environment setup
  - Handle API key configuration
  - Launch the Streamlit application

### Technology Details

- **LangChain**: Framework for the RAG pipeline and Chroma integration
  - `langchain` - Core functionality
  - `langchain_chroma` - Vector database integration
  - `langchain_google_genai` - Gemini model integration

- **Chroma Vector Database**: Stores profile embeddings for semantic search
  - Uses Gemini embedding model
  - Enables similarity-based candidate retrieval

- **Google Gemini AI**: Provides embedding and generation capabilities
  - `models/embedding-001` - For creating embeddings
  - `gemini-2.0-flash-lite` - For query responses

- **Streamlit**: Powers the web interface with interactive components

- **BeautifulSoup & Pandas**: For data processing and extraction

## Data Sources

This project uses mock LinkedIn profiles from:
https://hamed-alikhani.github.io/mock-linkedin-profiles/

The data is automatically downloaded and processed when you first run the application.

## Key Features

- **Enhanced UI**: Beautiful presentation of candidates and results
- **Visual Skills Tags**: Skills displayed as visual tags for better readability
- **Database Visualization**: Interactive charts showing profile statistics
- **Customizable Results**: Control how many matches to retrieve
- **Debug Mode**: Toggle to see raw AI responses for troubleshooting
- **Improved Error Handling**: Better handling of edge cases
- **Database Status Indicators**: Visual indicators of system status
- **Refresh Functionality**: Ability to refresh data status without restarting

## License

MIT

## Acknowledgements

- Thank you to the AI-Austin community for support and feedback
