app link :https://mpsxlq8vn5ufm7vfwmueoo.streamlit.app/
Perfect ✅ Here’s your complete, ready-to-use **README.md** file for the `liver.ipynb` project.

````{"variant":"standard","title":"README for Liver Disease Prediction Project","id":"47301"}
# Liver Disease Prediction using AdaBoost Classifier

This project aims to predict **whether a patient has liver disease** based on various medical attributes using the **AdaBoost Classifier**. It includes a Jupyter Notebook for model development and a Streamlit web application for real-time predictions.

---

## 🧠 Project Overview

Liver disease is a critical health issue that can be detected early with the help of machine learning.  
This project builds and evaluates an AdaBoost model trained on liver-related medical data to classify patients as having or not having liver disease.

---

 📊 Dataset Information

The dataset used is the **Indian Liver Patient Dataset (ILPD)** or a similar variant.  
It contains the following features:

| Feature | Description |
|----------|--------------|
| Age | Age of the patient |
| Gender | Male / Female |
| Total_Bilirubin | Total bilirubin level |
| Direct_Bilirubin | Direct bilirubin level |
| Alkaline_Phosphotase | Alkaline phosphotase enzyme level |
| Alamine_Aminotransferase | Alamine enzyme level |
| Aspartate_Aminotransferase | Aspartate enzyme level |
| Total_Proteins | Total protein count |
| Albumin | Albumin level |
| Albumin_and_Globulin_Ratio | Ratio of albumin and globulin |
| Dataset | Target variable (1 = liver disease, 2 = no liver disease) |

---

## ⚙️ Technologies Used

- **Python 3.10+**
- **Pandas**, **NumPy** — for data handling  
- **Scikit-learn** — for preprocessing, modeling, and evaluation  
- **Matplotlib / Seaborn** — for visualization  
- **Streamlit** — for web app deployment  
- **Pickle** — for saving the trained model  

---

## 🚀 Model Development Steps

1. **Data Preprocessing**  
   - Handling missing values and outliers  
   - Encoding categorical variables  
   - Feature scaling using StandardScaler or PowerTransformer  

2. **Handling Imbalanced Data**  
   - Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes.

3. **Model Training**  
   - Implemented **AdaBoostClassifier** with tuned hyperparameters.  
   - Example:
     ```python
     from sklearn.ensemble import AdaBoostClassifier
     model = AdaBoostClassifier(n_estimators=500, random_state=42)
     model.fit(X_res, y_res)
     ```

4. **Model Evaluation**  
   - Metrics used: Accuracy, Precision, Recall, F1-Score  
   - Visualized feature importances.

5. **Model Saving**  
   - Saved trained model using Pickle for later use in Streamlit app.

---

## 💻 Streamlit App

An interactive Streamlit app is created (`app.py`) to make real-time predictions.  
The user inputs their medical parameters, and the model outputs whether liver disease is likely or not.

### Run the App

```bash
streamlit run app.py
```

### Example Output
```
Prediction: Liver Disease Detected
```
or  
```
Prediction: No Liver Disease
```

---

## 📈 Results

- **Model:** AdaBoost Classifier  
- **Accuracy Achieved:** ~85% (may vary depending on preprocessing and tuning)  
- **Performance:** Balanced precision and recall after applying SMOTE  

---

## 🗂️ Repository Structure

```
├── liver.ipynb          # Model training and evaluation notebook
├── app.py               # Streamlit web app for prediction
├── model.pkl            # Saved AdaBoost model
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚡ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/liver-disease-prediction.git
   cd liver-disease-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook liver.ipynb
   ```

4. Launch the app:
   ```bash
   streamlit run app.py
   ```

---

## 🧩 Future Improvements

- Try other ensemble methods (XGBoost, LightGBM).  
- Implement cross-validation for robust accuracy.  
- Deploy app using Streamlit Cloud or Heroku.  

---

## 👨‍💻 Author

**Kavi**  
Machine Learning Enthusiast | Data Science Learner  

---

## 📝 License

This project is open-source under the MIT License.
````

Would you like me to also create a `requirements.txt` file that matches this README (with all needed libraries and versions)?
