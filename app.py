import streamlit as st
import pickle
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import xgboost as xgb

with open('xg_model.pkl', 'rb') as file:
    saved_data = joblib.load(file)
    loaded_model = saved_data['xg_model']
    loaded_scaler = saved_data['scaler']

used_df = pd.read_csv('used_in_model_training.csv')

st.markdown("""
    <h1 style='text-align: center; color: white;'>Used Cars Price Prediction</h1>
""", unsafe_allow_html=True)

selected_manufacturer = st.selectbox("Select Manufacturer", sorted(used_df['Manufacturer'].unique()))

selected_manufacturer_encoded = used_df[used_df['Manufacturer'] == selected_manufacturer]['Manufacturer_encoded'].iloc[0]

if selected_manufacturer == 'Other':
    selected_model_options = ["Other"]
else:
    selected_models = used_df[used_df['Manufacturer_encoded'] == selected_manufacturer_encoded]['Model'].unique()
    selected_model_options = ["Other"] + sorted(selected_models)

selected_model = st.selectbox("Select Model", selected_model_options)

if selected_model == "Other":
    selected_model_encoded = 695
else:
    selected_model_encoded = used_df[(used_df['Manufacturer_encoded'] == selected_manufacturer_encoded) & (used_df['Model'] == selected_model)]['Model_encoded'].iloc[0]

categories_1 = ['Coupe', 'Goods wagon', 'Hatchback', 'Microbus', 'Sedan']
categories_2 = ['Cabriolet', 'Jeep', 'Limousine', 'Minivan', 'Pickup', 'Universal']

if selected_manufacturer != 'Other' and selected_model != 'Other':
    category_from_dataset = used_df[(used_df['Manufacturer_encoded'] == selected_manufacturer_encoded) & (used_df['Model'] == selected_model)]['Category'].iloc[0]
    st.selectbox("Select Car Category", options=[category_from_dataset], index=0, key='disabled_category_selectbox')
    
    selected_category = category_from_dataset
    category_1 = 1 if selected_category in categories_1 else 0
    category_2 = 1 if selected_category in categories_2 else 0
else:
    selected_category = st.selectbox("Select Car Category", categories_1 + categories_2)
    category_1 = 1 if selected_category in categories_1 else 0
    category_2 = 1 if selected_category in categories_2 else 0

mileage = st.number_input("Enter Mileage (in km): ")



user_input_data = pd.DataFrame({'Mileage': [mileage]})
user_input_data['Mileage'] = loaded_scaler.transform(user_input_data[['Mileage']])
normalized_mileage = user_input_data['Mileage'].iloc[0]

production_year = st.number_input("Enter Production Year:", min_value=1900, max_value=2023, value=2022)
age = 2023 - production_year

engine_volume = st.slider('Select Engine Volume', min_value=0.8, max_value=4.0, value=2.0, step=0.1)
has_turbo = st.radio("Does the engine have a turbo?", ["Yes", "No"])
has_turbo_encoded = 1 if has_turbo == "Yes" else 0

selected_cylinders = st.slider("Select the number of cylinders", 1, 14, 4)
number_of_airbags = st.slider("Select Number of Airbags", min_value=0, max_value=16, value=0)

gear_box_types = ['Automatic', 'Tiptronic', 'Manual', 'Variator']
selected_gear_box_type = st.selectbox("Select Gear Box Type", gear_box_types)

gear_box_type_1 = 1 if selected_gear_box_type in ['Automatic', 'Variator'] else 0
gear_box_type_2 = 1 if selected_gear_box_type in ['Manual', 'Tiptronic'] else 0

drive_side_heading = st.radio("Select Drive Side Heading", ["Left Hand Drive", "Right Hand Drive"])
left_hand_drive = 1 if drive_side_heading == "Left Hand Drive" else 0
right_hand_drive = 1 if drive_side_heading == "Right Hand Drive" else 0

doors_2_3 = 0
doors_4_5 = 0
doors_greater_than_5 = 0

selected_doors = st.selectbox("Select Number of Doors", ['2-3', '4-5', '>5'])

if selected_doors == '2-3':
    doors_2_3 = 1
    doors_4_5 = 0
    doors_greater_than_5 = 0
elif selected_doors == '4-5':
    doors_2_3 = 0
    doors_4_5 = 1
    doors_greater_than_5 = 0
else:  # '>5'
    doors_2_3 = 0
    doors_4_5 = 0
    doors_greater_than_5 = 1

fuel_types = ['CNG', 'Diesel', 'LPG', 'Petrol', 'Hybrid', 'Hydrogen', 'Plug-in Hybrid']
selected_fuel_type = st.selectbox("Select Fuel Type", fuel_types)

fuel_type_petrol = 0
fuel_type_cng = 0
fuel_type_diesel = 0
fuel_type_lpg = 0
fuel_type_other = 0

if selected_fuel_type == 'Petrol':
    fuel_type_petrol = 1
elif selected_fuel_type == 'CNG':
    fuel_type_cng = 1
elif selected_fuel_type == 'Diesel':
    fuel_type_diesel = 1
elif selected_fuel_type == 'LPG':
    fuel_type_lpg = 1
else:
    fuel_type_other = 1

if st.button("Predict Price", key="predict_button", help="Click to predict the car price"):
    
    user_input = {
        'Engine volume':[engine_volume],
        'Mileage':[normalized_mileage],
        'Cylinders':selected_cylinders,
        'Airbags':[number_of_airbags],
        'Manufacturer_encoded':[selected_manufacturer_encoded],
        'Model_encoded':[selected_model_encoded],
        'Has Turbo':[has_turbo_encoded],
        'Age':[age],
        'Category_encoded_1':[category_1],
        'Category_encoded_2':[category_2],
        'Fuel type_CNG':[fuel_type_cng],
        'Fuel type_Diesel':[fuel_type_diesel],
        'Fuel type_LPG':[fuel_type_lpg],
        'Fuel type_Other':[fuel_type_other],
        'Fuel type_Petrol':[fuel_type_petrol],
        'Gear box type_1':[gear_box_type_1],
        'Gear box type_2':[gear_box_type_2],
        'Doors_2-3':[doors_2_3], 
        'Doors_4-5':[doors_4_5],
        'Doors_>5':[doors_greater_than_5], 
        'Wheel_Left wheel':[left_hand_drive], 
        'Wheel_Right-hand drive':[right_hand_drive]  
    }
    user_input_df = pd.DataFrame(user_input)
    user_input_dmatrix = xgb.DMatrix(data=user_input_df)

    predictions = loaded_model.predict(user_input_dmatrix)
    st.markdown(f"<p style='font-size:24px; text-align:center;'>Predicted Price : $ {predictions} </p>", unsafe_allow_html=True)
