"""
Text preprocessing module.
Single unified preprocessing function used throughout the entire pipeline.
"""
import re
from typing import Optional


def preprocess_text(text: str, remove_urls: bool = True, remove_emails: bool = True) -> str:
    """
    Unified text preprocessing function.
    Used in training, validation, test, and production pipelines.
    
    Args:
        text: Input text to preprocess
        remove_urls: Whether to remove URLs (default: True)
        remove_emails: Whether to remove email addresses (default: True)
    
    Returns:
        Preprocessed text string
    
    Steps:
        1. Convert to lowercase
        2. Remove special characters (keep Portuguese accents)
        3. Normalize whitespace
        4. Remove URLs and emails (optional)
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove URLs (if enabled)
    if remove_urls:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
    
    # Step 3: Remove emails (if enabled)
    if remove_emails:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '', text)
    
    # Step 4: Remove special characters but keep Portuguese accents
    # Keep: letters (including accented), numbers, spaces
    # Remove: punctuation and other special chars
    text = re.sub(r'[^\w\sáàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]', ' ', text)
    
    # Step 5: Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Step 6: Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_batch(texts: list[str], remove_urls: bool = True, remove_emails: bool = True) -> list[str]:
    """
    Batch preprocessing for multiple texts.
    
    Args:
        texts: List of input texts
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove emails
    
    Returns:
        List of preprocessed texts
    """
    return [preprocess_text(text, remove_urls, remove_emails) for text in texts]

