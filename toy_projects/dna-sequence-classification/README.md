# DNA Sequence Classification using Machine Learning

This project aims to develop a machine learning model that can classify DNA sequences into different categories based on their properties or functions.

## Project Steps

### 1. Data Collection
- Dataset: Choose a dataset that contains DNA sequences along with their corresponding categories or labels. For example, you can search for publicly available repositories like the NCBI (National Center for Biotechnology Information) or Kaggle for suitable datasets.

### 2. Data Preprocessing
- Import the dataset using a suitable library like Pandas to read the dataset into a DataFrame.
```python
import pandas as pd
df = pd.read_csv("DNA_sequences.csv")
```
- Clean the data by removing any missing values or artifacts in the DNA sequences.
```python
df.dropna(inplace=True)  # Drop rows with missing values
```
- Convert DNA sequences to numerical representation using one-hot encoding.
```python
# One-hot encoding function
def one_hot_encoding(sequence):
    encoding = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
    encoded_sequence = [encoding[nucleotide] for nucleotide in sequence]
    return encoded_sequence

# Apply one-hot encoding to the DNA sequences
df["Encoded_sequence"] = df["DNA_sequence"].apply(one_hot_encoding)
```

### 3. Feature Extraction
- Calculate the nucleotide frequency for each DNA sequence.
```python
# Calculate nucleotide frequency
df["Nucleotide_frequency"] = df["DNA_sequence"].apply(lambda x: {
    "A": x.count("A") / len(x),
    "C": x.count("C") / len(x),
    "G": x.count("G") / len(x),
    "T": x.count("T") / len(x)
})
```

### 4. Model Selection
- Choose a machine learning algorithm. For example, let's select the Random Forest classifier.
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100)
```

### 5. Model Training and Evaluation
- Split the dataset into training and testing sets.
```python
from sklearn.model_selection import train_test_split

# Split the dataset into features (X) and labels (y)
X = df["Nucleotide_frequency"].tolist()
y = df["Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Train the model.
```python
# Train the Random Forest classifier
classifier.fit(X_train, y_train)
```
- Evaluate the model's performance.
```python
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
```

### 6. Hyperparameter Tuning (Optional)
- Perform hyperparameter tuning to optimize the model's performance. For example, you can use GridSearchCV to search for the best combination of hyperparameters.
```python
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    "n_estimators": [50

, 100, 200],
    "max_depth": [None, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and retrain the model
best_params = grid_search.best_params_
best_classifier = RandomForestClassifier(**best_params)
best_classifier.fit(X_train, y_train)

# Evaluate the best model
y_pred_best = best_classifier.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Model Accuracy:", accuracy_best)
```

### 7. Model Deployment and Testing
- Deploy the trained model on new, unseen DNA sequences for predictions.
```python
new_sequences = ["ATCGATCG", "CGATCGAT", "TTAGCTAA"]
new_sequences_encoded = [one_hot_encoding(seq) for seq in new_sequences]
predictions = best_classifier.predict(new_sequences_encoded)
print("Predictions:", predictions)
```


