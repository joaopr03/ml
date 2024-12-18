import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import joblib
import pandas as pd

scaler = joblib.load("scaler.pkl")  # Pre-fitted scaler for all features
encoder = joblib.load("encoder.pkl")  # Pre-fitted encoder for categorical features
model = joblib.load("model.pkl")  # Trained model

def predictor():
    st.title("Claim Injury Type Prediction Interface")
    
    age_at_injury = st.number_input("Age at Injury:", min_value=16, max_value=75)
    attorney_rep = st.selectbox("Attorney/Representative (YES - 1 / NO - 0):", [0, 1])
    birth_year = st.number_input("Birth Year:", min_value=1930, max_value=2024, step=1)
    first_hearing = st.selectbox("First Hearing (YES - 1 / NO - 0):", [0, 1])
    c3_delivery = st.selectbox("C-3 Delivery (YES - 1 / NO - 0):", [0, 1])
    log_avg_weekly_wage_input = st.number_input("Log Average Weekly Wage (Input Raw Value):", min_value=0.0)
    log_avg_weekly_wage = np.log(log_avg_weekly_wage_input) if log_avg_weekly_wage_input >= 1 else 0.0
    forms_delivered_count = st.number_input("Forms Delivered Count:", min_value=0, max_value=10, step=1)
    valid_full_claim = st.selectbox("Valid Full Claim (YES - 1 / NO - 0):", [0, 1])
    assembly_year = st.number_input("Assembly Year:", min_value=1950, max_value=2024, step=1)
    carrier_name = st.text_input("Carrier Name:")
    county_of_injury = st.selectbox("County of Injury:", [
        'CAYUGA', 'QUEENS', 'MONROE', 'ALBANY', 'KINGS', 'WESTCHESTER',
        'JEFFERSON', 'NEW YORK', 'BROOME', 'NASSAU', 'CHEMUNG', 'BRONX',
        'SUFFOLK', 'DUTCHESS', 'ERIE', 'CLINTON', 'CHAUTAUQUA', 'ORANGE',
        'ONEIDA', 'RICHMOND', 'FULTON', 'MONTGOMERY', 'CATTARAUGUS',
        'ULSTER', 'SARATOGA', 'RENSSELAER', 'ONONDAGA', 'ROCKLAND',
        'SENECA', 'OSWEGO', 'DELAWARE', 'ST. LAWRENCE', 'NIAGARA', 'ESSEX',
        'WASHINGTON', 'ONTARIO', 'OTSEGO', 'WARREN', 'ORLEANS', 'PUTNAM',
        'ALLEGANY', 'HERKIMER', 'MADISON', 'WYOMING', 'SCHENECTADY',
        'LIVINGSTON', 'LEWIS', 'FRANKLIN', 'HAMILTON', 'SULLIVAN', 'WAYNE',
        'COLUMBIA', 'TOMPKINS', 'STEUBEN', 'CORTLAND', 'GREENE', 'TIOGA',
        'GENESEE', 'SCHUYLER', 'CHENANGO', 'SCHOHARIE', 'YATES'
    ])
    industry_code = st.number_input("Industry Code:", step=1)
    wcio_cause_code = st.selectbox("WCIO Cause of Injury Code:", [
        53.0, 31.0, 56.0, 32.0, 29.0, 83.0, 74.0, 60.0, 81.0, 89.0, 5.0,
        99.0, 75.0, 45.0, 68.0, 12.0, 30.0, 16.0, 88.0, 87.0, 33.0, 55.0,
        58.0, 76.0, 27.0, 28.0, 79.0, 97.0, 10.0, 19.0, 4.0, 57.0, 25.0,
        54.0, 20.0, 77.0, 80.0, 11.0, 52.0, 98.0, 2.0, 9.0, 70.0, 46.0,
        26.0, 85.0, 13.0, 48.0, 3.0, 82.0, 78.0, 15.0, 50.0, 90.0, 40.0,
        6.0, 18.0, 84.0, 7.0, 59.0, 95.0, 65.0, 69.0, 61.0, 1.0, 67.0,
        41.0, 86.0, 47.0, 96.0, 14.0, 91.0, 93.0, 8.0
    ])
    wcio_nature_code = st.selectbox("WCIO Nature of Injury Code:", [
        52.0, 10.0, 49.0, 90.0, 83.0, 59.0, 4.0, 16.0, 28.0, 43.0, 25.0,
        46.0, 40.0, 37.0, 13.0, 1.0, 34.0, 7.0, 80.0, 69.0, 36.0, 31.0,
        32.0, 72.0, 78.0, 77.0, 71.0, 53.0, 42.0, 2.0, 19.0, 55.0, 91.0,
        47.0, 73.0, 38.0, 68.0, 65.0, 58.0, 67.0, 3.0, 41.0, 75.0, 74.0,
        66.0, 61.0, 54.0, 70.0, 30.0, 60.0, 64.0, 62.0, 79.0, 63.0, 22.0,
        76.0
    ])
    wcio_part_code = st.selectbox("WCIO Part of Body Code:", [
        55.0, 65.0, 38.0, 32.0, 90.0, 60.0, 56.0, 10.0, 39.0, 35.0, 31.0,
        53.0, 15.0, 9.0, 18.0, 61.0, 33.0, 22.0, 34.0, 58.0, 11.0, 91.0,
        54.0, 99.0, 36.0, 44.0, 42.0, 14.0, 46.0, 13.0, 37.0, 51.0, 52.0,
        66.0, 30.0, 40.0, 12.0, 41.0, 57.0, 48.0, 20.0, 49.0, 17.0, 19.0,
        24.0, 23.0, 45.0, 50.0, 16.0, 63.0, 21.0, 62.0, 26.0, 64.0
    ])
    body_part_group = st.selectbox("Body Part Group:", [
        'Lower Extremities', 'Other', 'Upper Extremities', 'Lungs & Heart', 
        'Head & Neck', 'Torso', 'Spinal Cord', 'Buttocks'
    ])
    
    if st.button("Predict"):

        # List of all expected features
        all_features = ['Age at Injury', 'Alternative Dispute Resolution',
                        'Attorney/Representative', 'Average Weekly Wage', 'Birth Year',
                        'COVID-19 Indicator', 'Gender', 'IME-4 Count', 'First Hearing',
                        'C-3 Delivery', 'Days from Accident to C-2',
                        'Days from Accident to Assembly', 'Log_Average_Weekly_Wage',
                        'Claim Antiguity', 'C-2 under Deadline', 'Forms Delivered Count',
                        'Valid Full Claim', 'Accident Year', 'Accident Month', 'Assembly Year',
                        'Assembly Month', 'C-2 Year', 'C-2 Month', 'Carrier Name',
                        'Carrier Type', 'County of Injury', 'District Name', 'Industry Code',
                        'Medical Fee Region', 'WCIO Cause of Injury Code',
                        'WCIO Nature of Injury Code', 'WCIO Part Of Body Code', 'Zip Code',
                        'Age_Group', 'Injury Group', 'Body Part Group',
                        'Nature of Injury Group', 'Industry Group']

        count_encode = ['Carrier Name', 'Carrier Type', 'County of Injury',
                'District Name', 'Industry Code', 'Medical Fee Region',
                'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code',
                'WCIO Part Of Body Code', 'Zip Code', 'Age_Group',
                'Injury Group', 'Body Part Group', 'Nature of Injury Group',
                'Industry Group']
        
        feature_selection = ['Age at Injury',
                    'Attorney/Representative',
                    'Birth Year',
                    'First Hearing',
                    'C-3 Delivery',
                    'Log_Average_Weekly_Wage',
                    'Forms Delivered Count',
                    'Valid Full Claim',
                    'Assembly Year',
                    'Carrier Name',
                    'County of Injury',
                    'Industry Code',
                    'WCIO Cause of Injury Code',
                    'WCIO Nature of Injury Code',
                    'WCIO Part Of Body Code',
                    'Body Part Group']
        
        input_data = {
            'Age at Injury': age_at_injury,
            'Attorney/Representative': attorney_rep,
            'Birth Year': birth_year,
            'First Hearing': first_hearing,
            'C-3 Delivery': c3_delivery,
            'Log_Average_Weekly_Wage': log_avg_weekly_wage,
            'Forms Delivered Count': forms_delivered_count,
            'Valid Full Claim': valid_full_claim,
            'Assembly Year': assembly_year,
            'Carrier Name': carrier_name if carrier_name else "Unknown",
            'County of Injury': county_of_injury,
            'Industry Code': industry_code,
            'WCIO Cause of Injury Code': wcio_cause_code,
            'WCIO Nature of Injury Code': wcio_nature_code,
            'WCIO Part Of Body Code': wcio_part_code,
            'Body Part Group': body_part_group
        }

        # Fill missing features with placeholder values
        complete_data = {feature: 0 for feature in all_features}  # Initialize with default values
        complete_data.update(input_data)  # Update with user inputs

        input_df = pd.DataFrame([complete_data])

        expected_columns = input_df.columns.tolist()
        
        # Fill missing values
        input_df[count_encode] = input_df[count_encode].fillna("Unknown")
        numerical_columns = [col for col in all_features if col not in count_encode]
        input_df[numerical_columns] = input_df[numerical_columns].fillna(0)

        encoded_categorical = encoder.transform(input_df[count_encode])   

        # Combine encoded categorical and numerical features
        combined_df = pd.concat([encoded_categorical, input_df[numerical_columns]], axis=1)
        combined_df = combined_df[expected_columns]

        scaled_data = scaler.transform(combined_df)

        # Convert scaled_data back to a DataFrame with the original column names
        scaled_df = pd.DataFrame(scaled_data, columns=combined_df.columns)
        
        predict_data = scaled_df[feature_selection]
        prediction = model.predict(predict_data)

        prediction_series = pd.Series(prediction)

        prediction_series = prediction_series.replace({
            1: '1. CANCELLED',
            2: '2. NON-COMP',
            3: '3. MED ONLY',
            4: '4. TEMPORARY',
            5: '5. PPD SCH LOSS',
            6: '6. PPD NSL',
            7: '7. PTD',
            8: '8. DEATH'
        })

        prediction = prediction_series.values
        st.success(f"The prediction is: {prediction[0]}")

def map_page():
    st.title('Discounts')
    st.write('Check out our special discounts and offers!')
    
    # Define the coordinates for New York and its four regions (GeoJSON data)
    nyc_coords = [40.771720, -73.890148]  # Center of New York City
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Region 1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-74.016, 40.705],
                            [-73.975, 40.705],
                            [-73.975, 40.74],
                            [-74.016, 40.74],
                            [-74.016, 40.705],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Region 2"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-74.005, 40.74],
                            [-73.965, 40.74],
                            [-73.965, 40.77],
                            [-74.005, 40.77],
                            [-74.005, 40.74],
                        ]
                    ],
                },
            },
            {
            "type": "Feature",
            "properties": {"name": "Region 3"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-73.975, 40.77],
                        [-73.935, 40.77],
                        [-73.935, 40.8],
                        [-73.975, 40.8],
                        [-73.975, 40.77]
                    ]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {"name": "Region 4"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-73.945, 40.8],
                        [-73.905, 40.8],
                        [-73.905, 40.83],
                        [-73.945, 40.83],
                        [-73.945, 40.8]
                    ]
                ]
            }
        },
        ],
    }
    
    # Read the data from the file
    try:
        with open('open_ended_regions.txt', 'r') as file:
            lines = file.readlines()
            r1_data = lines[0].strip().split(",") if len(lines) > 0 else ["No data for r1"]
            r2_data = lines[1].strip().split(",") if len(lines) > 1 else ["No data for r2"]
            r3_data = lines[2].strip().split(",") if len(lines) > 2 else ["No data for r3"]
            r4_data = lines[3].strip().split(",") if len(lines) > 3 else ["No data for r4"]
    except FileNotFoundError:
        st.error("The file 'open_ended_regions.txt' was not found.")
    
    # Create a layout with two columns
    col1, col2 = st.columns([2, 1])  # Adjust column widths as needed (map takes 2/3, text takes 1/3)
    
    # Display the map in the left column
    with col1:
        m = folium.Map(location=nyc_coords, zoom_start=12)
        folium.GeoJson(
            geojson_data,
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Region"]),
        ).add_to(m)
        
        # Adjust map size
        map_output = st_folium(m, width=800, height=550)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display the clicked region details in the right column
    with col2:
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: flex-start">
            <h3>Details</h3>
            """,
            unsafe_allow_html=True,
        )
        if map_output and map_output.get("last_active_drawing", None):
            clicked_feature = map_output["last_active_drawing"]['properties']
            region_name = clicked_feature.get("name", "Unknown Region")
            
            st.markdown(f"<strong>Region Name:</strong> {region_name}", unsafe_allow_html=True)
            
            # Show data for the selected region
            try:
                if region_name == "Region 1":
                    st.markdown(f"Age at Injury Min: {r1_data[0]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Mean: {r1_data[1]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Max: {r1_data[2]}", unsafe_allow_html=True)
                    st.markdown(f"Gender: {r1_data[3]}", unsafe_allow_html=True)
                    st.markdown(f"District Name: {r1_data[4]}", unsafe_allow_html=True)
                    st.markdown(f"Injury Group: {r1_data[5]}", unsafe_allow_html=True)
                    st.markdown(f"Body Part Group: {r1_data[6]}", unsafe_allow_html=True)
                    st.markdown(f"Industry Group: {r1_data[7]}", unsafe_allow_html=True)
                elif region_name == "Region 2":
                    st.markdown(f"Age at Injury Min: {r2_data[0]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Mean: {r2_data[1]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Max: {r2_data[2]}", unsafe_allow_html=True)
                    st.markdown(f"Gender: {r2_data[3]}", unsafe_allow_html=True)
                    st.markdown(f"District Name: {r2_data[4]}", unsafe_allow_html=True)
                    st.markdown(f"Injury Group: {r2_data[5]}", unsafe_allow_html=True)
                    st.markdown(f"Body Part Group: {r2_data[6]}", unsafe_allow_html=True)
                    st.markdown(f"Industry Group: {r2_data[7]}", unsafe_allow_html=True)
                elif region_name == "Region 3":
                    st.markdown(f"Age at Injury Min: {r3_data[0]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Mean: {r3_data[1]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Max: {r3_data[2]}", unsafe_allow_html=True)
                    st.markdown(f"Gender: {r3_data[3]}", unsafe_allow_html=True)
                    st.markdown(f"District Name: {r3_data[4]}", unsafe_allow_html=True)
                    st.markdown(f"Injury Group: {r3_data[5]}", unsafe_allow_html=True)
                    st.markdown(f"Body Part Group: {r3_data[6]}", unsafe_allow_html=True)
                    st.markdown(f"Industry Group: {r3_data[7]}", unsafe_allow_html=True)
                elif region_name == "Region 4":
                    st.markdown(f"Age at Injury Min: {r4_data[0]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Mean: {r4_data[1]}", unsafe_allow_html=True)
                    st.markdown(f"Age at Injury Max: {r4_data[2]}", unsafe_allow_html=True)
                    st.markdown(f"Gender: {r4_data[3]}", unsafe_allow_html=True)
                    st.markdown(f"District Name: {r4_data[4]}", unsafe_allow_html=True)
                    st.markdown(f"Injury Group: {r4_data[5]}", unsafe_allow_html=True)
                    st.markdown(f"Body Part Group: {r4_data[6]}", unsafe_allow_html=True)
                    st.markdown(f"Industry Group: {r4_data[7]}", unsafe_allow_html=True)
                else:
                    st.markdown("No data available for this region.", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.markdown("Click on a region to see details.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
def home_page():
    st.markdown("## Our group members")
    st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <p>Afonso Gamito, 20240752</p>
            <p>João Rodrigues, 20241037</p>
            <p>Rute D’Alva Teixeira, 20240667</p>
            <p>Samuel Mendes, 20240751</p>
            <p>Tomás Oliveira, 20211576</p>
        </div>
    """, unsafe_allow_html=True)

    # GitHub button
    st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://github.com/joaopr03/ml" target="_blank" style="text-decoration: none;">
                <button style="
                    display: inline-flex;
                    align-items: center;
                    background-color: #333;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 20px;
                    font-size: 16px;
                    cursor: pointer;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16" style="margin-right: 8px;">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/>
                    </svg>
                    Group 8
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# Main function that controls the navigation
def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Choose a page", ['Home', 'Predictor Claim Injury Type', 'Regions Map'])

    if selection == 'Home':
        home_page()
    elif selection == 'Predictor Claim Injury Type':
        predictor()
    elif selection == 'Regions Map':
        map_page()

if __name__ == '__main__':
    main()
