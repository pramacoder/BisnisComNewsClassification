import pandas as pd
import re
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def clean_text(text):
    """
    Cleans the input text by removing URLs, special characters, and extra spaces.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits (keeping only letters)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_resampling(X, y, method='smote', random_state=42):
    """
    Applies resampling to handle imbalanced data.
    
    Parameters:
    - X: Feature matrix (typically sparse matrix from TF-IDF for text)
    - y: Target labels
    - method: 'smote', 'under', or 'combined'
    
    Returns:
    - X_resampled, y_resampled
    """
    if method == 'smote':
        resampler = SMOTE(random_state=random_state)
    elif method == 'under':
        resampler = RandomUnderSampler(random_state=random_state)
    elif method == 'combined':
        # Over-sample minority classes to 50% of majority, then under-sample majority to match
        # (This is just an example strategy, usually configured via sampling_strategy dictionaries)
        over = SMOTE(random_state=random_state)
        under = RandomUnderSampler(random_state=random_state)
        resampler = Pipeline(steps=[('o', over), ('u', under)])
    else:
        raise ValueError("Invalid method. Choose 'smote', 'under', or 'combined'")
        
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    return X_resampled, y_resampled

if __name__ == '__main__':
    # Simple test for clean_text
    sample = "Baca selengkapnya di https://bisnis.com. Saham #BBCA naik 2.5% hari ini!"
    print(f"Original: {sample}")
    print(f"Cleaned:  {clean_text(sample)}")
