import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.title("🔄 Telecom Customer Churn Predictor")
st.markdown("Enter customer details below to predict if they will cancel their subscription, and see the **SHAP explanation** of why.")

# Load model and columns (cached for performance)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model/churn_model.pkl')
        model_cols = joblib.load('model/model_columns.pkl')
        return model, model_cols
    except FileNotFoundError:
        return None, None

model, model_columns = load_assets()

if model is None:
    st.error("⚠️ Model files not found! Please run `python model.py` first to train and save the model.")
    st.stop()

# --- User Input Layout ---
st.sidebar.header("Customer Profile")

def get_user_input():
    # Core predictive features available via UI
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    MonthlyCharges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
    TotalCharges = st.sidebar.slider("Total Charges ($)", 18.0, 8600.0, MonthlyCharges * tenure if tenure > 0 else 18.0)
    
    Contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    InternetService = st.sidebar.selectbox("Internet Service", ['Fiber optic', 'DSL', 'No'])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    
    # Create a dictionary. We hardcode remaining minor features to their baseline to keep the UI clean.
    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'Contract': Contract,
        'InternetService': InternetService,
        'PaymentMethod': PaymentMethod,
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'PaperlessBilling': 'Yes'
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# Display input summary
st.subheader("Current Customer Data")
st.dataframe(input_df.style.format(precision=2))

# --- Data Processing ---
# One-hot encode the user input
input_encoded = pd.get_dummies(input_df)

# Reindex to ensure it matches the exact columns the model was trained on
# fill_value=0 ensures missing categorical dummy columns are set to 0
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# --- Prediction & Explainability ---
if st.button("🔮 Predict Churn Risk"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"**High Risk of Churn!**")
            st.write(f"Probability of leaving: **{probability[1] * 100:.1f}%**")
        else:
            st.success(f"**Customer is likely to Stay.**")
            st.write(f"Probability of staying: **{probability[0] * 100:.1f}%**")

    with col2:
        st.subheader("Why? (SHAP Feature Contributions)")
        with st.spinner("Calculating SHAP values..."):
          # Initialize SHAP Tree Explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_encoded)
            
            # Handle different SHAP library versions dynamically
            if isinstance(shap_values, list):
                shap_val_churn = shap_values[1][0]          # Older SHAP versions
            elif len(shap_values.shape) == 3:
                shap_val_churn = shap_values[0, :, 1]       # Newer SHAP versions (3D array)
            else:
                shap_val_churn = shap_values[0]             # Fallback (2D array)
            
            # Create a dataframe for visualization
            shap_df = pd.DataFrame({
                'Feature': model_columns,
                'Impact': shap_val_churn
            })
            
            # Get top 10 most impactful features for THIS specific prediction
            shap_df['Abs_Impact'] = shap_df['Impact'].abs()
            top_shap = shap_df.sort_values(by='Abs_Impact', ascending=False).head(10)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['red' if x > 0 else 'green' for x in top_shap['Impact']]
            sns.barplot(data=top_shap, x='Impact', y='Feature', palette=colors, ax=ax)
            ax.set_title("Factors Driving This Prediction (Red = Promotes Churn, Green = Prevents Churn)")
            ax.set_xlabel("SHAP Value (Impact on Model Output)")
            st.pyplot(fig)