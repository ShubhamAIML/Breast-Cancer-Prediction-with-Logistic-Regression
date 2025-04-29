import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    roc_curve, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tabulate import tabulate

# Function to plot the sigmoid function (used in logistic regression)
def plot_sigmoid():
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(x, y, label='Sigmoid Function', color='blue')
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('σ(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig('sigmoid_plot.png')
    plt.show()
    plt.close()

# Function to plot confusion matrix with percentages and save the figure
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum() * 100
    plt.figure(figsize=(8, 6), dpi=100)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False)

    # Annotate each cell with count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5,
                     f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)',
                     ha='center', va='center', color='black')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5], ['Benign', 'Malignant'])
    plt.yticks([0.5, 1.5], ['Benign', 'Malignant'])
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()
    plt.close()

    return {'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1]}


# Function to plot the ROC curve
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.show()
    plt.close()


# Load the breast cancer dataset
data = pd.read_csv('data.csv')

# Drop unnecessary columns
columns_to_drop = ['id']
if 'Unnamed: 32' in data.columns:
    columns_to_drop.append('Unnamed: 32')
data = data.drop(columns=columns_to_drop)

# Check for essential column
if 'fractal_dimension_worst' not in data.columns:
    raise ValueError("Expected column 'fractal_dimension_worst' not found in dataset.")

# Encode target variable: Malignant (M) = 1, Benign (B) = 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Display missing values
print("Missing Values in Dataset:")
print(data.isnull().sum().to_string())
print()

# Fill missing values using mean imputation
data = data.fillna(data.mean(numeric_only=True))

# Save the cleaned and preprocessed data
data.to_csv('processed_data.csv', index=False)

X = data.drop('diagnosis', axis=1)  # Features
y = data['diagnosis']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler for future use
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Compute evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Print classification report
print("Classification Report (Threshold=0.5):")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
print()

# Display metrics in table format
metrics_table = [
    ["Metric", "Value"],
    ["Precision", f"{precision:.2f}"],
    ["Recall", f"{recall:.2f}"],
    ["ROC-AUC", f"{auc:.2f}"]
]
print("Model Metrics (Threshold=0.5):")
print(tabulate(metrics_table, headers='firstrow', tablefmt='grid'))
print()

# Plot confusion matrix
cm_metrics = plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix (Threshold=0.5)')
print("Confusion Matrix Metrics (Threshold=0.5):")
print(tabulate([
    ["True Negatives (TN)", cm_metrics['TN']],
    ["False Positives (FP)", cm_metrics['FP']],
    ["False Negatives (FN)", cm_metrics['FN']],
    ["True Positives (TP)", cm_metrics['TP']]
], headers=['Metric', 'Value'], tablefmt='grid'))
print()

# Plot ROC curve
plot_roc_curve(y_test, y_prob)

# Plot sigmoid function for conceptual understanding
plot_sigmoid()

thresholds = [0.3, 0.5, 0.7]

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)

    print(f"\nThreshold: {thresh}")
    metrics_table = [
        ["Metric", "Value"],
        ["Precision", f"{precision:.2f}"],
        ["Recall", f"{recall:.2f}"]
    ]
    print(tabulate(metrics_table, headers='firstrow', tablefmt='grid'))

    cm_metrics = plot_confusion_matrix(y_test, y_pred_thresh, f'Confusion Matrix (Threshold={thresh})')
    print("Confusion Matrix Metrics:")
    print(tabulate([
        ["True Negatives (TN)", cm_metrics['TN']],
        ["False Positives (FP)", cm_metrics['FP']],
        ["False Negatives (FN)", cm_metrics['FN']],
        ["True Positives (TP)", cm_metrics['TP']]
    ], headers=['Metric', 'Value'], tablefmt='grid'))
    print()

print("Model training complete.")
print("Saved: model.joblib, scaler.joblib, processed_data.csv")
print("Visualizations saved: sigmoid_plot.png, roc_curve.png, confusion_matrix_*.png")
