# Breast Cancer Prediction with Logistic Regression 🩺💍  

Welcome to the **Breast Cancer Prediction** project! This repository contains a machine learning model built using **Logistic Regression** and a **Flask web application** to predict breast cancer diagnosis (*Malignant* or *Benign*) based on the **Breast Cancer Wisconsin Dataset**.  

**[🌐Live Web App](https://breast-cancer-prediction-u9xj.onrender.com/)**  |  **[📓Google Colab Notebook](https://colab.research.google.com/drive/1qbyuTQ7OY-8u-GZx-tzhuzd1pndlnsuf?usp=sharing)**   |  **[📓Kaggle Notebook](https://www.kaggle.com/code/shubham1921/notebook7054d49d75)**

## 📖 **Overview**  
The goal of this project is to:  
- Build a **binary classifier** using **Logistic Regression** to predict breast cancer diagnosis.  
- Evaluate the model using metrics like **confusion matrix, precision, recall, and ROC-AUC**.  
- Create a **user-friendly Flask web app** for real-time predictions.  
- Provide **clear documentation** and a **reproducible setup**.  

The Flask app allows users to input features and receive predictions, styled with **custom CSS** for a polished look. 🎨  

---

## 🗂️ **Folder Structure**  
```
breast_cancer_prediction/  
├── static/  
│   └── css/  
│       └── style.css       # Custom CSS for web app styling 🎨  
├── templates/  
│   ├── index.html          # Input form page for user inputs 📝  
│   └── result.html         # Results page displaying predictions ✅  
├── app.py                  # Flask application for predictions 🌐  
├── confusion_matrix_(threshold=0.3).png  # Confusion matrix at threshold 0.3 📊  
├── confusion_matrix_(threshold=0.5).png  # Confusion matrix at threshold 0.5 📊  
├── confusion_matrix_(threshold=0.7).png  # Confusion matrix at threshold 0.7 📊  
├── data.csv                # Breast Cancer Wisconsin Dataset 📊  
├── model.joblib            # Trained Logistic Regression model 🤖  
├── processed_data.csv      # Preprocessed dataset 📊  
├── requirements.txt        # Python dependencies ⚙️  
├── roc_curve.png           # ROC curve plot 📈  
├── scaler.joblib           # StandardScaler for feature scaling ⚖️  
├── sigmoid_plot.png        # Sigmoid function plot 📈  
├── task_4.pdf              # Task description document 📄  
├── train_model.py          # Script to train and save the model 🏋️  
└── README.md               # Project documentation (you're here!) 📖  
```  

---

## 📊 **Dataset**  
The project uses the **Breast Cancer Wisconsin Dataset** from Kaggle. This dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses, with the target variable indicating whether the diagnosis is:  
- **Malignant (M)**: Cancerous  
- **Benign (B)**: Non-cancerous  

**Key details:**  
- **Source**: Kaggle  
- **Features**: 30 numerical features (e.g., radius, texture, perimeter)  
- **Target**: Binary (Malignant or Benign)  
- **File**: `data.csv` in the repository  
- **Preprocessed File**: `processed_data.csv` contains the dataset after cleaning and encoding  

---

## 🛠️ **Tools and Libraries**  
The project is built using the following tools and libraries:  
- **Python 3.8+** 🐍: Programming language  
- **Scikit-learn** 🤖: For Logistic Regression, feature scaling, and evaluation metrics  
- **Pandas** 📊: Data manipulation and preprocessing  
- **Matplotlib** 📈: Visualization of results (e.g., ROC curve, confusion matrix, sigmoid plot)  
- **Flask** 🌐: Web framework for the prediction app  
- **Joblib** 💾: Saving and loading the trained model and scaler  
- **NumPy** 🔢: Numerical computations  
- **HTML/CSS** 🎨: Frontend for the Flask app  
- **GitHub** 🗄️: Version control and submission  

See `requirements.txt` for the full list of dependencies.  

---

## 🚀 **How It Works**  

### **1. Model Training (`train_model.py`) 🏋️**  
- **Dataset Loading**: Loads the Breast Cancer Wisconsin Dataset from `data.csv`.  
- **Preprocessing**:  
  - Drops unnecessary columns (e.g., ID).  
  - Encodes the target variable (M → 1, B → 0).  
  - Saves preprocessed data to `processed_data.csv`.  
  - Splits data into training and test sets (80:20 ratio).  
  - Standardizes features using `StandardScaler`.  
- **Model**: Trains a **Logistic Regression** model using Scikit-learn.  
- **Evaluation**:  
  - Computes **confusion matrix, precision, recall, and ROC-AUC**.  
  - Plots the **ROC curve (`roc_curve.png`)** and **sigmoid function (`sigmoid_plot.png`)**.  
  - Generates **confusion matrices** at thresholds **0.3, 0.5, and 0.7** (`confusion_matrix_(threshold=0.X).png`).  
- **Saving**: Saves the trained model (`model.joblib`) and scaler (`scaler.joblib`).  

### **2. Flask Web App (`app.py`) 🌐**  
- **Frontend**:  
  - `index.html`: A form to input 30 feature values.  
  - `result.html`: Displays the prediction (*Malignant* or *Benign*) with a confidence score.  
  - `style.css`: Custom styling for a clean and modern UI.  
- **Backend**:  
  - Loads the trained model and scaler.  
  - Processes user inputs, scales them, and predicts the diagnosis.  
  - Returns the result with the probability (sigmoid output).  

### **3. Key Concepts Covered 📚**  
- **Logistic Regression**: Uses the sigmoid function to predict probabilities for binary classification.  
- **Sigmoid Function**: Maps input values to a range (0, 1) for probability estimation (see `sigmoid_plot.png`).  
- **Evaluation Metrics**:  
  - **Confusion Matrix**: Visualized at different thresholds (see `confusion_matrix_(threshold=0.X).png`).  
  - **Precision**: Accuracy of positive predictions.  
  - **Recall**: Ability to identify all positive cases.  
  - **ROC-AUC**: Measures model performance across thresholds (see `roc_curve.png`).  
- **Threshold Tuning**: Evaluated at thresholds **0.3, 0.5, and 0.7** to balance precision and recall.  
- **Imbalanced Classes**: Addressed by evaluating metrics like recall and ROC-AUC.  

---

## 📈 **Results**  
The **Logistic Regression model** achieves:  
- **Accuracy**: ~95% (varies based on train/test split)  
- **Precision**: High for both classes  
- **Recall**: High, especially for Malignant cases (critical for medical diagnosis)  
- **ROC-AUC**: ~0.98, indicating excellent model performance (see `roc_curve.png`)  

The **Flask app** provides a seamless way to interact with the model, delivering predictions in real-time. 🕒  

---

## 🏃‍♂️ **How to Run the Project**  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/breast_cancer_prediction.git  
   cd breast_cancer_prediction  
   ```  

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt  
   ```  

3. **Train the Model** (optional, as `model.joblib` and `scaler.joblib` are included):  
   ```bash
   python train_model.py  
   ```  

4. **Run the Flask App**:  
   ```bash
   python app.py  
   ```  

5. **Open your browser** and go to `http://127.0.0.1:5000` to access the app.  

6. **Interact with the App**:  
   - Enter the **30 feature values** in the form.  
   - Submit to see the **prediction and confidence score**.  

---

## 📝 **Interview Questions Covered**  
The project addresses the following questions:  
1. **How does logistic regression differ from linear regression?**  
   - Logistic regression predicts probabilities for binary outcomes using the **sigmoid function**, while linear regression predicts continuous values.  
2. **What is the sigmoid function?**  
   - A mathematical function that maps inputs to (0, 1), used to compute probabilities (visualized in `sigmoid_plot.png`).  
3. **What is precision vs recall?**  
   - **Precision** is the ratio of correct positive predictions to total positive predictions.  
   - **Recall** is the ratio of correct positive predictions to all actual positives.  
4. **What is the ROC-AUC curve?**  
   - A plot of **True Positive Rate vs False Positive Rate**, with AUC indicating model performance (see `roc_curve.png`).  
5. **What is the confusion matrix?**  
   - A table showing **true positives, true negatives, false positives, and false negatives** (visualized in `confusion_matrix_(threshold=0.X).png`).  
6. **What happens if classes are imbalanced?**  
   - The model may favor the majority class, requiring metrics like recall or techniques like resampling.  
7. **How do you choose the threshold?**  
   - By balancing precision and recall based on the application (evaluated at thresholds **0.3, 0.5, and 0.7**).  
8. **Can logistic regression be used for multi-class problems?**  
   - Yes, using techniques like **One-vs-Rest** or **Softmax Regression**.  

---

## 🔗 **Submission Details**  
- **GitHub Repository**: This repository contains all code, datasets, visualizations, and documentation.  
- **Task**: Completed as per the **Elevate AI & ML Internship Task 4** guidelines (see `task_4.pdf`).
- **Time Window**: Task completed within the allowed window (10:00 AM to 10:00 PM).  

---

## 🙌 **Acknowledgments**  
- **Elevate**: For providing this learning opportunity.  
- **Kaggle**: For the **Breast Cancer Wisconsin Dataset**.  
- **Open-Source Community**: For tools like Scikit-learn, Flask, and Pandas.  

Feel free to explore the code, run the app, and provide feedback! If you have any questions, reach out via **GitHub Issues**. 🌟  

**Happy Coding!** 🚀
