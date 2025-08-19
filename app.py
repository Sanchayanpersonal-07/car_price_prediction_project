import streamlit as st
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Custom CSS for a Dark Theme
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        padding: 20px;
        background: #1E1E1E;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    h1, h2, h3 {
        color: #BB86FC;
        text-align: center;
    }
    .stSidebar > div {
        padding: 20px;
        background: #1E1E1E;
        border-radius: 10px;
        color: #FFFFFF;
    }
    .stTextInput>div>label,
    .stRadio>div>label,
    .stNumberInput>div>label,
    .stSlider>div>label {
        color: #BB86FC;
    }
    .stButton>button {
        background-color: #BB86FC;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3700B3;
    }
    .output-box {
        background: #292929;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🚗 Car Price Predictor")

# Tabs for Navigation
tabs = st.tabs(["Home", "About"])

with tabs[0]:
    # Instructions
    st.markdown("""
    Predict the resale value of a used car based on key features.  
    Enter the car details in the **sidebar** and click **Predict** to see the result.
    """)

    # Sidebar Inputs
    st.sidebar.header("Enter Car Details")
    car_name = st.sidebar.text_input("Car Name", help="E.g., Toyota, Honda")
    vehicle_age = st.sidebar.number_input("Vehicle Age (Years)", 0, 20, value=5, step=1)
    kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, value=10000, step=500)
    seller_type = st.sidebar.radio("Seller Type", ("Dealer", "Individual"))
    fuel_type = st.sidebar.radio("Fuel Type", ("Petrol", "Diesel", "CNG"))
    transmission_type = st.sidebar.radio("Transmission Type", ("Manual", "Automatic"))
    mileage = st.sidebar.number_input("Mileage (km/l)", 5.0, 40.0, value=18.0, step=0.5)
    engine = st.sidebar.number_input("Engine Capacity (CC)", 500, 5000, value=1500, step=100)
    max_power = st.sidebar.number_input("Max Power (BHP)", 20, 500, value=100, step=5)
    seats = st.sidebar.slider("Number of Seats", 2, 8, value=5)

    # Load the model
    try:
        model = pickle.load(open("rf_regressor.pkl", "rb"))
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'rf_regressor.pkl' is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

    # Prediction
    if st.sidebar.button("Predict"):
        input_data = np.array([[vehicle_age, kms_driven, 
                                1 if seller_type == "Individual" else 0, 
                                1 if fuel_type == "Diesel" else (2 if fuel_type == "CNG" else 0), 
                                1 if transmission_type == "Automatic" else 0, 
                                mileage, engine, max_power, seats]])
        try:
            prediction = model.predict(input_data)
            st.markdown("<div class='output-box'>", unsafe_allow_html=True)
            st.write(f"💰 **Estimated Selling Price:** ₹{prediction[0]:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tabs[1]:
    # About Section
    st.subheader("About This App")
    st.markdown("""
    **Car Price Predictor** is a machine learning-powered application designed to help users estimate the resale value of used cars.  
    It leverages advanced algorithms to analyze multiple parameters like the vehicle's age, mileage, engine specifications, and more.
    
    ### Key Features:
    - **Easy-to-use Interface**: Input car details and get instant predictions.
    - **Accurate Estimations**: Powered by a robust predictive model.
    - **Customizable Options**: Choose fuel type, seller type, and more.
                

    #### Built with:
    - **Python**
    - **Streamlit**
    - **Scikit-learn**
    - **Numpy and Pandas**

    For any queries or feedback, please reach out to the developer.


    This project was created as part of an innovation-driven machine learning initiative. 🚀
    """)
