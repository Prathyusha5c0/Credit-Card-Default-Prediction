import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model_path = 'logistic_regression_model.pkl'    
loaded_model = joblib.load(model_path)
def preprocess_input(input_data):
    sex_mapping = {'Male': 1, 'Female': 2}
    education_mapping = {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4}
    marriage_mapping = {'Others': 3, 'Married': 1, 'Single': 2, 'Others': 3}
    pay_status_mapping = {
        'No consumption': -1, 'Paid in full': 0, 'Use of revolving credit': 0,
        'Payment delay for one month': 1, 'Payment delay for two months': 2,
        'Payment delay for three months': 3, 'Payment delay for four months': 4,
        'Payment delay for five months': 5, 'Payment delay for six months': 6,
        'Payment delay for seven months': 7, 'Payment delay for eight months': 8,
        'Payment delay for nine months and above': 9
    }
    processed_input = {
        'SEX': sex_mapping.get(input_data['SEX'], 1),
        'EDUCATION': education_mapping.get(input_data['EDUCATION'], 1),
        'MARRIAGE': marriage_mapping.get(input_data['MARRIAGE'], 1),
        'AGE': input_data['AGE'],
        'LIMIT_BAL': input_data['LIMIT_BAL'],
        'PAY_4': pay_status_mapping.get(input_data['PAY_4'], 0),
        'PAY_5': pay_status_mapping.get(input_data['PAY_5'], 0),
        'PAY_6': pay_status_mapping.get(input_data['PAY_6'], 0),
        'BILL_AMT4': input_data['BILL_AMT4'],
        'BILL_AMT5': input_data['BILL_AMT5'],
        'BILL_AMT6': input_data['BILL_AMT6'],
        'PAY_AMT4': input_data['PAY_AMT4'],
        'PAY_AMT5': input_data['PAY_AMT5'],
        'PAY_AMT6': input_data['PAY_AMT6'],
    }

    processed_df = pd.DataFrame([processed_input])

    return processed_df

def predict_default(input_data):
    processed_input = preprocess_input(input_data)
    df = pd.DataFrame(processed_input)
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    df_cat = df[categorical_features]
    df_cat.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 'EDUCATION' : {1 : 'graduate school', 2 : 'university', 3 : 'high school', 4 : 'others'}, 'MARRIAGE' : {1 : 'married', 2 : 'single', 3 : 'others'}}, inplace = True)

    df.rename(columns={'PAY_4':'PAY_JUN','PAY_5':'PAY_MAY','PAY_6':'PAY_APR'},inplace=True)
    df.rename(columns={'BILL_AMT4':'BILL_AMT_JUN','BILL_AMT5':'BILL_AMT_MAY','BILL_AMT6':'BILL_AMT_APR'}, inplace = True)
    df.rename(columns={'PAY_AMT4':'PAY_AMT_JUN','PAY_AMT5':'PAY_AMT_MAY','PAY_AMT6':'PAY_AMT_APR'},inplace=True)

    df['AGE']=df['AGE'].astype('int')

    df_fr = df.copy()
    df_fr.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 'EDUCATION' : {1 : 'graduate school', 2 : 'university', 3 : 'high school', 4 : 'others'}, 'MARRIAGE' : {1 : 'married', 2 : 'single', 3 : 'others'}}, inplace = True)

    df_fr = pd.get_dummies(df_fr,columns=['EDUCATION','MARRIAGE'])
    column_names = ['ID', 'LIMIT_BAL', 'SEX', 'AGE', 'BILL_AMT_JUN', 'BILL_AMT_MAY',
                'BILL_AMT_APR', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR',
                'default.payment.next.month', 'EDUCATION_graduate school',
                'EDUCATION_high school', 'EDUCATION_university', 'MARRIAGE_married',
                'MARRIAGE_single', 'PAY_JUN_-1', 'PAY_JUN_0', 'PAY_JUN_1', 'PAY_JUN_2',
                'PAY_JUN_3', 'PAY_JUN_4', 'PAY_JUN_5', 'PAY_JUN_6', 'PAY_JUN_7',
                'PAY_JUN_8', 'PAY_MAY_-1', 'PAY_MAY_0', 'PAY_MAY_2', 'PAY_MAY_3',
                'PAY_MAY_4', 'PAY_MAY_5', 'PAY_MAY_6', 'PAY_MAY_7', 'PAY_MAY_8',
                'PAY_APR_-1', 'PAY_APR_0', 'PAY_APR_2', 'PAY_APR_3', 'PAY_APR_4',
                'PAY_APR_5', 'PAY_APR_6', 'PAY_APR_7', 'PAY_APR_8']

    missing_columns = set(column_names) - set(df_fr.columns)
    for col in missing_columns:
      df_fr[col] = 0  
    df_fr = pd.get_dummies(df_fr, columns = ['PAY_JUN','PAY_MAY','PAY_APR'], drop_first = True )
    encoders_nums = {
                 "SEX":{"FEMALE": 0, "MALE": 1}
}
    df_fr = df_fr.replace(encoders_nums)
    df_log_reg = df_fr.copy()
    df_log_reg.head()
    df_fr = df_fr[column_names]
    df_fr = df_fr.replace(encoders_nums)
    df_log_reg = df_fr.copy()
    prediction = loaded_model.predict(df_log_reg)
    return prediction




def main():
    st.title('Credit Card Default Prediction')
    st.sidebar.header('Input Fields')
    id_input = st.sidebar.number_input('ID', value=1, step=1)
    limit_bal_input = st.sidebar.number_input('LIMIT_BAL', value=0)
    sex_input = st.sidebar.selectbox('SEX', ['Male', 'Female'])
    education_input = st.sidebar.selectbox('EDUCATION', ['Graduate School', 'University', 'High School', 'Others'])
    marriage_input = st.sidebar.selectbox('MARRIAGE', ['Others', 'Married', 'Single', 'Others'])
    age_input = st.sidebar.number_input('AGE', value=18, min_value=18, max_value=100, step=1)
    pay_status_input_jun = st.sidebar.selectbox('Repayment Status in June (PAY_4)', ['No consumption', 'Paid in full', 'Use of revolving credit', 'Payment delay for one month', 'Payment delay for two months', 'Payment delay for three months', 'Payment delay for four months', 'Payment delay for five months', 'Payment delay for six months', 'Payment delay for seven months', 'Payment delay for eight months', 'Payment delay for nine months and above'])
    pay_status_input_may = st.sidebar.selectbox('Repayment Status in May (PAY_5)', ['No consumption', 'Paid in full', 'Use of revolving credit', 'Payment delay for one month', 'Payment delay for two months', 'Payment delay for three months', 'Payment delay for four months', 'Payment delay for five months', 'Payment delay for six months', 'Payment delay for seven months', 'Payment delay for eight months', 'Payment delay for nine months and above'])
    pay_status_input_apr = st.sidebar.selectbox('Repayment Status in April (PAY_6)', ['No consumption', 'Paid in full', 'Use of revolving credit', 'Payment delay for one month', 'Payment delay for two months', 'Payment delay for three months', 'Payment delay for four months', 'Payment delay for five months', 'Payment delay for six months', 'Payment delay for seven months', 'Payment delay for eight months', 'Payment delay for nine months and above'])
    bill_statement_jun = st.sidebar.number_input('Amount of bill statement in June', value=0)
    bill_statement_may = st.sidebar.number_input('Amount of bill statement in May', value=0)
    bill_statement_apr = st.sidebar.number_input('Amount of bill statement in April', value=0)
    prev_payment_jun = st.sidebar.number_input('Amount of previous payment in June', value=0)
    prev_payment_may = st.sidebar.number_input('Amount of previous payment in May', value=0)
    prev_payment_apr = st.sidebar.number_input('Amount of previous payment in April', value=0)


    if st.sidebar.button('Submit'):
        user_inputs = {
            'ID': id_input,
            'LIMIT_BAL': limit_bal_input,
            'SEX': sex_input,
            'EDUCATION': education_input,
            'MARRIAGE': marriage_input,
            'AGE': age_input,
            'PAY_4': pay_status_input_jun,
            'PAY_5': pay_status_input_may,
            'PAY_6': pay_status_input_apr,
            'BILL_AMT4': bill_statement_jun,
            'BILL_AMT5': bill_statement_may,
            'BILL_AMT6': bill_statement_apr,
            'PAY_AMT4': prev_payment_jun,
            'PAY_AMT5': prev_payment_may,
            'PAY_AMT6': prev_payment_apr,
        }

        prediction = predict_default(loaded_model, user_inputs)

        st.write('### Prediction:')
        if prediction[0] == 0:
            st.write('Not a Defaulter')
        else:
            st.write('Defaulter')
    st.write('### Input Values:')
    st.write(f'ID: {id_input}')
    st.write(f'LIMIT_BAL: {limit_bal_input}')
    st.write(f'SEX: {sex_input}')
    st.write(f'EDUCATION: {education_input}')
    st.write(f'MARRIAGE: {marriage_input}')
    st.write(f'AGE: {age_input}')
    st.write(f'PAY_4: {pay_status_input_jun}')
    st.write(f'PAY_5: {pay_status_input_may}')
    st.write(f'PAY_6: {pay_status_input_apr}')
    st.write(f'Amount of bill statement in June: {bill_statement_jun}')
    st.write(f'Amount of bill statement in May: {bill_statement_may}')
    st.write(f'Amount of bill statement in April: {bill_statement_apr}')
    st.write(f'Amount of previous payment in June: {prev_payment_jun}')
    st.write(f'Amount of previous payment in May: {prev_payment_may}')
    st.write(f'Amount of previous payment in April: {prev_payment_apr}')

def main():
    st.title('Credit Card Default Prediction')
    st.sidebar.header('Input Fields')
    id_input = st.sidebar.number_input('ID', value=1, step=1)
    limit_bal_input = st.sidebar.number_input('LIMIT_BAL', value=0)
    sex_input = st.sidebar.selectbox('SEX', ['Male', 'Female'])
    education_input = st.sidebar.selectbox('EDUCATION', ['Graduate School', 'University', 'High School', 'Others'])
    marriage_input = st.sidebar.selectbox('MARRIAGE', ['Others', 'Married', 'Single', 'Others'])
    age_input = st.sidebar.number_input('AGE', value=18, min_value=18, max_value=100, step=1)
    pay_status_input_jun = st.sidebar.selectbox('Repayment Status in June (PAY_4)', ['No consumption', 'Paid in full', 'Use of revolving credit', 'Payment delay for one month', 'Payment delay for two months', 'Payment delay for three months', 'Payment delay for four months', 'Payment delay for five months', 'Payment delay for six months', 'Payment delay for seven months', 'Payment delay for eight months', 'Payment delay for nine months and above'])
    pay_status_input_may = st.sidebar.selectbox('Repayment Status in May (PAY_5)', ['No consumption', 'Paid in full', 'Use of revolving credit', 'Payment delay for one month', 'Payment delay for two months', 'Payment delay for three months', 'Payment delay for four months', 'Payment delay for five months', 'Payment delay for six months', 'Payment delay for seven months', 'Payment delay for eight months', 'Payment delay for nine months and above'])
    pay_status_input_apr = st.sidebar.selectbox('Repayment Status in April (PAY_6)', ['No consumption', 'Paid in full', 'Use of revolving credit', 'Payment delay for one month', 'Payment delay for two months', 'Payment delay for three months', 'Payment delay for four months', 'Payment delay for five months', 'Payment delay for six months', 'Payment delay for seven months', 'Payment delay for eight months', 'Payment delay for nine months and above'])
    bill_statement_jun = st.sidebar.number_input('Amount of bill statement in June', value=0)
    bill_statement_may = st.sidebar.number_input('Amount of bill statement in May', value=0)
    bill_statement_apr = st.sidebar.number_input('Amount of bill statement in April', value=0)
    prev_payment_jun = st.sidebar.number_input('Amount of previous payment in June', value=0)
    prev_payment_may = st.sidebar.number_input('Amount of previous payment in May', value=0)
    prev_payment_apr = st.sidebar.number_input('Amount of previous payment in April', value=0)


    if st.sidebar.button('Submit'):
        user_inputs = {
            'ID': id_input,
            'LIMIT_BAL': limit_bal_input,
            'SEX': sex_input,
            'EDUCATION': education_input,
            'MARRIAGE': marriage_input,
            'AGE': age_input,
            'PAY_4': pay_status_input_jun,
            'PAY_5': pay_status_input_may,
            'PAY_6': pay_status_input_apr,
            'BILL_AMT4': bill_statement_jun,
            'BILL_AMT5': bill_statement_may,
            'BILL_AMT6': bill_statement_apr,
            'PAY_AMT4': prev_payment_jun,
            'PAY_AMT5': prev_payment_may,
            'PAY_AMT6': prev_payment_apr,
        }

        prediction = predict_default(user_inputs)

        st.write('### Prediction:')
        if prediction[0] == 0:
            st.write('Not a Defaulter')
        else:
            st.write('Defaulter')

    # Display input values
    st.write('### Input Values:')
    st.write(f'ID: {id_input}')
    st.write(f'LIMIT_BAL: {limit_bal_input}')
    st.write(f'SEX: {sex_input}')
    st.write(f'EDUCATION: {education_input}')
    st.write(f'MARRIAGE: {marriage_input}')
    st.write(f'AGE: {age_input}')
    st.write(f'PAY_4: {pay_status_input_jun}')
    st.write(f'PAY_5: {pay_status_input_may}')
    st.write(f'PAY_6: {pay_status_input_apr}')
    st.write(f'Amount of bill statement in June: {bill_statement_jun}')
    st.write(f'Amount of bill statement in May: {bill_statement_may}')
    st.write(f'Amount of bill statement in April: {bill_statement_apr}')
    st.write(f'Amount of previous payment in June: {prev_payment_jun}')
    st.write(f'Amount of previous payment in May: {prev_payment_may}')
    st.write(f'Amount of previous payment in April: {prev_payment_apr}')

main()