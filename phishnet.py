import pandas as pd
import os
import re
import subprocess
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def prepare_dataset():
    datasets = []
    files = {
        "CEAS_08.csv": 1,
        "Enron.csv": 0,
        "Ling.csv": 0,
        "Nazario.csv": 1,
        "Nigerian_Fraud.csv": 1,
        "phishing_email.csv": 1,
        "SpamAssasin.csv": 1
    }
    for file, label in files.items():
        if os.path.exists(file):
            df = pd.read_csv(file, encoding='latin1')
            df['label'] = label
            datasets.append(df)
    combined_df = pd.concat(datasets, ignore_index=True)
    if 'text' in combined_df.columns and 'body' not in combined_df.columns:
        combined_df['body'] = combined_df['text']
    if 'email' in combined_df.columns and 'body' not in combined_df.columns:
        combined_df['body'] = combined_df['email']
    combined_df = combined_df[['body', 'label']]
    combined_df.dropna(inplace=True)
    combined_df = combined_df[combined_df['body'].str.strip() != '']
    combined_df['body'] = combined_df['body'].apply(clean_text)
    combined_df.to_csv("cleaned_emails.csv", index=False)
    print(f"[✓] Combined {len(combined_df)} emails saved to cleaned_emails.csv")

def simulate_phishing_email():
    print("\n[!] Simulating phishing email with swaks...")
    print("[!] Using Attacker: yuvindeakin@gmail.com → Defender: deakindefender@gmail.com\n")
    recipient = input("Enter recipient email (default: deakindefender@gmail.com): ") or "deakindefender@gmail.com"
    subject = input("Enter email subject: ")
    body = input("Enter email body: ")
    cmd = [
        "swaks",
        "--to", recipient,
        "--from", "yuvindeakin@gmail.com",
        "--server", "localhost",
        "--data", f"Subject: {subject}\n\n{body}"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("[✓] Phishing email sent from yuvindeakin@gmail.com to", recipient)
    except Exception as e:
        print(f"[✗] Failed to send phishing email: {e}")

def transfer_to_defender():
    print("\n[~] Transferring cleaned_emails.csv to Defender VM...")
    defender_ip = "192.168.1.20"
    target_path = f"kali@{defender_ip}:/home/kali/Documents/TaskHD326-D/"
    try:
        subprocess.run(["scp", "cleaned_emails.csv", target_path], check=True)
        print("[✓] File transferred successfully to Defender VM!")
    except Exception as e:
        print(f"[✗] Transfer failed: {e}")

def train_and_evaluate_models():
    print("\n[~] Loading cleaned_emails.csv...")
    df = pd.read_csv("cleaned_emails.csv")
    X = df['body'].fillna('').astype(str)
    y = df['label']
    print("[~] Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    print("[~] Splitting data for training and testing...")
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    print("[~] Training Support Vector Machine...")
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("[~] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("\n=== SVM Evaluation ===")
    print(classification_report(y_test, y_pred_svm))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    print("\n=== Random Forest Evaluation ===")
    print(classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("rf_confusion_matrix.png")
    plt.show()
    print("[✓] Training complete. Evaluation report and confusion matrix saved.")

if __name__ == "__main__":
    print("\n--- PHISHNET+ MULTI-FUNCTION SCRIPT ---")
    print("1. Prepare Dataset")
    print("2. Simulate Phishing Email")
    print("3. Train and Evaluate ML Models")
    choice = input("Choose an option (1, 2, or 3): ")
    if choice == "1":
        prepare_dataset()
        transfer_to_defender()
    elif choice == "2":
        simulate_phishing_email()
    elif choice == "3":
        train_and_evaluate_models()
    else:
        print("Invalid choice.")
