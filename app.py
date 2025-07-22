import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="💰Income Classification", layout="wide")
st.title("Income Classification Prediction App")

model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")

categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
feature_order = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation', 'relationship',
                 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

st.subheader("Enter Individual Details")

age = st.number_input("Age", min_value=17, max_value=90, value=30)
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=200000)
educational_num = st.slider("Education Number", 1, 16, 10)

workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
                                       'State-gov', 'Without-pay', 'Never-worked'])
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                                                 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                         'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                         'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                         'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
gender = st.radio("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)
native_country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                                                  'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                                                  'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
                                                  'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
                                                  'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
                                                  'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
                                                  'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

for col in categorical_cols:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

# Fix column order
input_df = input_df[feature_order]

if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    prediction_label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {prediction_label}")

st.markdown("---")
st.subheader("Batch Prediction from CSV File")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    if 'education' in batch_data.columns:
        batch_data.drop(columns=['education'], inplace=True)

    batch_data.replace('?', 'Not-Listed', inplace=True)

    for col in categorical_cols:
        if col in batch_data.columns:
            batch_data[col] = encoders[col].transform(batch_data[col])

    batch_data = batch_data[feature_order]

    predictions = model.predict(batch_data)
    batch_data['Predicted Income'] = [">50K" if p == 1 else "<=50K" for p in predictions]
    st.write("Prediction Results:")
    st.dataframe(batch_data)