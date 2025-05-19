from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

# Experience model
class Experience(BaseModel):
    """Experience entry from a profile"""
    title: str
    company: str
    duration: str
    description: str
    
# Skill model (simple string for now, could be expanded)
class Skill(BaseModel):
    """Skill entry with optional endorsement count"""
    name: str
    endorsement_count: Optional[int] = None

# Profile model
class Profile(BaseModel):
    """LinkedIn-style profile model"""
    name: str
    headline: str
    contact_number: Optional[str] = ""
    available_spot: Optional[str] = ""
    city: Optional[str] = ""
    source_url: Optional[str] = ""
    experience: List[Union[Experience, Dict[str, Any]]] = []
    skills: List[str] = []
    about: Optional[str] = None
    raw_html: Optional[str] = None
    
    @field_validator('experience')
    def validate_experience(cls, experience_list):
        """Convert dict experiences to Experience objects if needed"""
        result = []
        for exp in experience_list:
            if isinstance(exp, dict):
                result.append(Experience(**exp))
            else:
                result.append(exp)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary without raw_html for cleaner output"""
        result = self.model_dump(exclude={"raw_html"})
        return result

# Candidate match model
class CandidateMatch(BaseModel):
    """Represents a matched candidate with score and reasoning"""
    profile: Profile
    match_score: int = Field(..., ge=1, le=100)  # Score 1-100%
    explanation: str
    relevant_skills: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output formatting"""
        return {
            "name": self.profile.name,
            "title": self.profile.headline,
            "score": f"{self.match_score}%",
            "explanation": self.explanation,
            "skills": self.relevant_skills,
            "contact": self.profile.contact_number
        }

# Search response model
class SearchResponse(BaseModel):
    """Structured response for a candidate search"""
    candidates: List[CandidateMatch]
    summary: str
    
    def to_formatted_response(self) -> str:
        """Convert the result to a formatted text response"""
        parts = []
        
        # Add each candidate
        for i, candidate in enumerate(self.candidates):
            candidate_dict = candidate.to_dict()
            parts.append(f"{i+1}. {candidate_dict['name']} ({candidate_dict['title']})")
            parts.append(f"   Match score: {candidate_dict['score']}")
            parts.append(f"   Explanation: {candidate_dict['explanation']}")
            parts.append(f"   Key skills: {', '.join(candidate_dict['skills'])}")
            parts.append(f"   Contact: {candidate_dict['contact']}")
            parts.append("")
        
        # Add summary but use a different format so it won't match regex patterns
        # that are looking for "Overall summary:"
        if self.summary:
            parts.append(f"SUMMARY: {self.summary}")
        
        return "\n".join(parts) 