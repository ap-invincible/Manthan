import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json

print("🚀 Starting Model Training Pipeline...\n")

# Step 1: Load Datasets
print("📂 Loading datasets...")
jobs_df = pd.read_csv('jobs_dataset.csv')
candidates_df = pd.read_csv('candidates_dataset.csv')
applications_df = pd.read_csv('applications_training_dataset.csv')

print(f"✅ Loaded {len(jobs_df)} jobs, {len(candidates_df)} candidates, {len(applications_df)} applications\n")

# Step 2: Prepare Training Data
print("⚙️ Preparing training data...")

# Features (X) and Target (y)
X = applications_df[['skill_match_score', 'experience_match_score', 'cgpa_score', 'projects_score']]
y = applications_df['outcome']

# Encode target labels (hired=2, shortlisted=1, rejected=0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Features shape: {X.shape}")
print(f"Target classes: {label_encoder.classes_}\n")

# Step 3: Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")

# Step 4: Train Multiple Models
print("🤖 Training Machine Learning Models...\n")

# Model 1: Random Forest
print("1️⃣ Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of decision trees
    max_depth=10,      # Maximum depth of trees
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"✅ Random Forest Accuracy: {rf_accuracy * 100:.2f}%\n")

# Model 2: Gradient Boosting
print("2️⃣ Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"✅ Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%\n")

# Step 5: Choose Best Model
print("🏆 Model Comparison:")
print(f"Random Forest: {rf_accuracy * 100:.2f}%")
print(f"Gradient Boosting: {gb_accuracy * 100:.2f}%\n")

if rf_accuracy >= gb_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_predictions = rf_pred
else:
    best_model = gb_model
    best_model_name = "Gradient Boosting"
    best_predictions = gb_pred

print(f"🥇 Best Model: {best_model_name}\n")

# Step 6: Model Evaluation
print("📊 Detailed Model Evaluation:\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, best_predictions))
print("\nClassification Report:")
print(classification_report(y_test, best_predictions, target_names=label_encoder.classes_))

# Feature Importance
print("\n🔍 Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# Step 7: Cross-Validation
print("\n🔄 Cross-Validation (5-fold):")
cv_scores = cross_val_score(best_model, X, y_encoded, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)\n")

# Step 8: Save Model and Encoders
print("💾 Saving trained model...")
joblib.dump(best_model, 'job_recommendation_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Save model metadata
metadata = {
    'model_type': best_model_name,
    'accuracy': float(rf_accuracy if best_model_name == "Random Forest" else gb_accuracy),
    'features': list(X.columns),
    'classes': list(label_encoder.classes_),
    'training_samples': len(X_train),
    'test_accuracy': float(accuracy_score(y_test, best_predictions))
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✅ Model saved as 'job_recommendation_model.pkl'")
print("✅ Label encoder saved as 'label_encoder.pkl'")
print("✅ Metadata saved as 'model_metadata.json'\n")

print("🎉 Training Complete! Your model is ready to use.\n")

# Step 9: Example Prediction
print("📝 Example Prediction:")
sample_data = pd.DataFrame({
    'skill_match_score': [75.0],
    'experience_match_score': [80.0],
    'cgpa_score': [85.0],
    'projects_score': [62.5]
})

prediction = best_model.predict(sample_data)
prediction_proba = best_model.predict_proba(sample_data)
predicted_class = label_encoder.inverse_transform(prediction)[0]

print(f"Input: {sample_data.values[0]}")
print(f"Predicted Outcome: {predicted_class}")
print(f"Confidence: {max(prediction_proba[0]) * 100:.2f}%")
print(f"Probabilities: {dict(zip(label_encoder.classes_, prediction_proba[0]))}")
