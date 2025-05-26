"""
Handler for blocked content detection and feedback generation.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import re
import logging
from enum import Enum

from app.services.sentiment.relevance_detection import RelevanceDetector, RelevanceScore, RelevanceLevel

@dataclass
class BlockedContentResult:
    """Result from blocked content analysis."""
    is_blocked: bool
    reason: str
    score: float
    details: Dict[str, Any]

class BlockedContentDetector:
    """Detects and provides feedback for blocked content."""
    
    def __init__(self):
        self.detector = RelevanceDetector()

    def is_blocked(self, text: str) -> BlockedContentResult:
        """
        Analyze text to determine if it should be blocked.
        
        Args:
            text: The text to analyze
            
        Returns:
            BlockedContentResult with analysis details
        """
        result = self.detector.analyze_text(text)
        
        # Block if not relevant
        if result.level in [RelevanceLevel.IRRELEVANT, RelevanceLevel.LOW]:
            return BlockedContentResult(
                is_blocked=True,
                reason="relevance",
                score=result.confidence,
                details={
                    "type": "irrelevant",
                    "evidence": result.supporting_evidence,
                    "explanation": result.explanation
                }
            )
            
        return BlockedContentResult(
            is_blocked=False,
            reason="",
            score=result.confidence,
            details={"type": "passed"}
        )

    def get_feedback_message(self, result: BlockedContentResult) -> str:
        """
        Generate helpful feedback message for blocked content.
        
        Args:
            result: The BlockedContentResult from content analysis
            
        Returns:
            str: User-friendly feedback message
        """
        if not result.is_blocked:
            return None
            
        if result.reason == "relevance":
            if result.details["type"] == "irrelevant":
                return "Content not sufficiently cryptocurrency-related. Try adding more specific cryptocurrency terms or context."
                
            return "Content needs more cryptocurrency-specific context. Try adding relevant cryptocurrency terms or technical details."
            
        return "Content needs more cryptocurrency-specific context. Try adding relevant cryptocurrency terms or technical details."

# Global detector instance
_detector = BlockedContentDetector()

def is_content_blocked(text: str) -> BlockedContentResult:
    """
    Check if content should be blocked.
    
    Args:
        text: The text to analyze
        
    Returns:
        BlockedContentResult with analysis details
    """
    return _detector.is_blocked(text) 