from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()

# Features and labels
X = iris["data"]
y = iris["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = KNeighborsClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Show prediction
print("Predicted Values:")
print(prediction)

# Show actual values
print("\nActual Values:")
print(y_test)

# Accuracy
accuracy = model.score(X_test, y_test)

print("\nAccuracy:", accuracy * 100, "%")