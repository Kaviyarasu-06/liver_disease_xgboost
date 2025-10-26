"""Train an XGBoost classifier on the Indian Liver Patient dataset and save a model artifact.

Saves a joblib file `liver_model.pkl` containing a dict with keys:
 - 'model': trained sklearn Pipeline
 - 'columns': list of feature column names in order

Run:
    python train_model.py
"""
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler


def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    # map target: 1 -> liver disease, 2 -> no liver disease (map to 0)
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
    # map Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df


def train(save_path: str = 'liver_model.pkl', csv_path: str = 'indian_liver_patient (1).csv'):
    df = load_and_prepare(csv_path)
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    artifact = {'model': pipeline, 'columns': X.columns.tolist()}
    joblib.dump(artifact, save_path)
    print(f"Saved trained model and metadata to {save_path}")


if __name__ == '__main__':
    train()
