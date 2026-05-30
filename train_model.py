import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def train_pipeline():
    print("[ML Pipeline] Loading the 2-class feature data...")
    
    # 1. Load the real data extracted from the images
    try:
        df = pd.read_csv("drowsiness_data.csv")
    except FileNotFoundError:
        print("Error: Could not find 'drowsiness_data.csv'.")
        return

    # Separate the features from the target label
    X = df[['ear', 'blink_duration', 'mar', 'head_deviation']]
    y = df['state'] 
    
    # --- PHASE 2 REQUIREMENT: Data Preprocessing ---
    print("[ML Pipeline] Applying StandardScaler to features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We MUST save this scaler so the Streamlit app can apply the exact same math to live webcam data
    joblib.dump(scaler, "feature_scaler.pkl")
    print("[ML Pipeline] Scaler saved successfully as 'feature_scaler.pkl'")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- PHASE 2 REQUIREMENT: Classical ML Setup ---
    model = xgb.XGBClassifier(
        n_estimators=1000,           # Set artificially high so early stopping can control the limit
        max_depth=5,
        learning_rate=0.05,
        objective='binary:logistic', # Optimized for our 2-class system
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=15
    )
    
    # --- PHASE 2 REQUIREMENT: Early Stopping ---
    print("[ML Pipeline] Training XGBoost model with Early Stopping...")
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=True
    )
    
    # Evaluate the final model's performance
    preds = model.predict(X_test)
    print("\n🚀 --- Binary Model Evaluation Summary ---")
    print(f"Overall Accuracy Score: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Focused', 'Fatigued']))
    
    # Save the optimized weights
    model.save_model("focus_model.json")
    print("[ML Pipeline] Production-ready model serialized to 'focus_model.json'")

if __name__ == "__main__":
    train_pipeline()