"""Streamlit app to predict liver disease using a saved model `liver_model.pkl`.

Usage:
    streamlit run app.py
"""
import streamlit as st
import joblib
import pandas as pd

MODEL_PATH = 'liver_model.pkl'

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        artifact = joblib.load(path)
        model = artifact.get('model') if isinstance(artifact, dict) else artifact
        columns = artifact.get('columns') if isinstance(artifact, dict) else None
        return model, columns
    except Exception as e:
        # If the model file is not present, try to infer columns from the
        # training CSV so the UI can still show input fields for manual
        # prediction (batch CSV upload remains optional).
        try:
            df = pd.read_csv('indian_liver_patient (1).csv')
            cols = [c for c in df.columns if c != 'Dataset']
            return None, cols
        except Exception:
            # Last resort: report the original error to the user and return None
            st.warning(f"Could not load model and could not infer columns: {e}")
            return None, None


def predict_row(model, columns, row_dict):
    df = pd.DataFrame([row_dict], columns=columns)
    pred = model.predict(df)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(df)[0][1])
    return int(pred), proba


def main():
    st.title('Liver Disease Prediction (XGBoost)')
    st.write('Provide patient features and click Predict')

    model, columns = load_model()
    if columns is None:
        st.warning('Model and dataset not found. Place `liver_model.pkl` or the CSV in the workspace.')
        return

    st.sidebar.header('Input features')
    # build inputs using column names from training
    inputs = {}
    for col in columns:
        if col == 'Gender':
            inputs[col] = st.sidebar.selectbox('Gender', options=['Male', 'Female'])
        else:
            # numeric inputs
            inputs[col] = st.sidebar.number_input(col, value=0.0)

    # coerce Gender to numeric
    if 'Gender' in inputs:
        inputs['Gender'] = 1 if str(inputs['Gender']).lower().startswith('m') else 0

    if st.sidebar.button('Predict single'):
        if model is None:
            st.error('No trained model available. Click "Train model" to train an XGBoost model using the bundled CSV.')
        else:
            pred, proba = predict_row(model, columns, inputs)
            st.write('Prediction (1 = disease, 0 = no disease):', pred)
            if proba is not None:
                st.write('Probability of disease:', round(proba, 4))

    # If model missing, offer to train from the UI
    if model is None:
        if st.sidebar.button('Train model'):
            with st.spinner('Training model — this may take a minute'):
                try:
                    # import the training module and invoke train()
                    import importlib
                    tm = importlib.import_module('train_model')
                    # call train and create liver_model.pkl in workspace
                    tm.train()
                    st.success('Training completed — reloading model')
                    # reload model
                    model, columns = load_model()
                except Exception as ex:
                    st.error(f'Error during training: {ex}')

    # Batch CSV upload has been removed per user request. Use the sidebar
    # single-prediction form to make predictions one patient at a time.
    st.markdown('---')
    st.info('Batch CSV upload disabled. Use the sidebar to enter individual patient features and click Predict single.')


if __name__ == '__main__':
    main()
