import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # We will use this to save the model for Day 2

# 1. Load the dataset
# Make sure the csv file is in the same folder as this script
data = pd.read_csv('Crop_recommendation.csv')

# Let's peek at the first 5 rows to ensure it loaded correctly
print("Data successfully loaded! Here is a preview:")
print(data.head())

# 2. Separate the Features (X) and the Target (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# 3. Split the data (80% training, 20% testing)
# random_state=42 just ensures we get the exact same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")

# 4. Initialize the Random Forest model
# n_estimators=100 means we are building 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using our 80% training data
print("\nTraining the AI model... (this might take a few seconds)")
rf_model.fit(X_train, y_train)

# 5. Make predictions on the hidden 20% of data
predictions = rf_model.predict(X_test)

# Calculate how many it got right
accuracy = accuracy_score(y_test, predictions)
print(f"\nBOOM! Model Training Complete.")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model so we can use it in our web app tomorrow
joblib.dump(rf_model, 'crop_model.pkl')
print("\nModel saved as 'crop_model.pkl'. Day 1 objective complete.")



