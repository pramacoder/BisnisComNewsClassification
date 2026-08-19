import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

from preprocessing import clean_text, apply_resampling
from evaluation import plot_confusion_matrix, plot_classification_report

def train_and_evaluate(data_path='data/train.csv', model_type='lr', use_smote=False):
    """
    Trains a baseline model handling imbalanced data via class_weight or SMOTE.
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    # Assume target column is 'label' and text column is 'text' based on previous inspection
    
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Stratified split to maintain class distribution in val set
    X_train, X_val, y_train, y_val = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    if use_smote:
        print("Applying SMOTE...")
        X_train_vec, y_train = apply_resampling(X_train_vec, y_train, method='smote')
        cw = None # don't use class weights if we already resampled
    else:
        cw = 'balanced'
        
    print(f"Training {model_type.upper()} model (class_weight={cw})...")
    if model_type == 'lr':
        clf = LogisticRegression(class_weight=cw, max_iter=1000, random_state=42)
    elif model_type == 'svm':
        clf = SVC(class_weight=cw, probability=True, random_state=42)
    elif model_type == 'rf':
        clf = RandomForestClassifier(class_weight=cw, random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'lr', 'svm', or 'rf'.")
        
    clf.fit(X_train_vec, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    classes = [str(c) for c in sorted(df['label'].unique())]
    
    # Save visualizations
    os.makedirs('outputs', exist_ok=True)
    plot_confusion_matrix(y_val, y_pred, classes=classes, normalize=True, save_path='outputs/cm_norm.png')
    plot_classification_report(y_val, y_pred, classes=classes, save_path='outputs/clf_report.png')
    
    # Try predicting probabilities for ROC/PR if applicable
    if hasattr(clf, "predict_proba"):
        from evaluation import plot_roc_curve, plot_pr_curve
        y_score = clf.predict_proba(X_val_vec)
        plot_roc_curve(y_val, y_score, classes=classes, save_path='outputs/roc_curve.png')
        plot_pr_curve(y_val, y_score, classes=classes, save_path='outputs/pr_curve.png')
        
    print("Training complete. Visualizations saved to 'outputs/' directory.")
    
if __name__ == '__main__':
    train_and_evaluate(model_type='lr', use_smote=False)
