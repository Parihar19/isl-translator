import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle # Used to save the AI's "brain" to a file

print("Loading data from CSV...")
# 1. Load the dataset
df = pd.read_csv('isl_dataset.csv')

# THE FIX FOR MESSY DATA: Automatically drop any rows that have blank spaces or missing numbers!
df = df.dropna()

# 2. Split into Features (X) and Labels (y)
y = df['label'] # This is the answer key (the letters A, B, C...)
X = df.drop('label', axis=1) # These are the 84 mathematical coordinates

# 3. Shuffle and split the data into "Training" and "Testing" piles
# We hide 20% of the data from the AI so we can give it a blind test later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the AI using a Random Forest algorithm
print("Training the AI model... (This might take a few seconds)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Give the AI a blind test on the 20% hidden data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print("Model Training Complete!")
print(f"AI Accuracy on hidden test data: {accuracy * 100:.2f}%")
print("-" * 30)

# 6. Save the trained AI model to a file so we can use it in our live camera
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("AI brain successfully saved as 'model.pkl'!")