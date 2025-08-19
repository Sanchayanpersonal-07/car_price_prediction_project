import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Load your dataset (replace with your actual dataset file path)
data = pd.read_csv("cardekho_dataset.csv")  # Replace with your data file

# 2. Preprocess data (example assumes your data has similar columns)
X = data[["vehicle_age", "km_driven", "fuel_type", "seller_type", "transmission_type", "mileage", "max_power", "engine", "seats"]]
y = data["selling_price"]

# Convert categorical columns to numeric (encoding)
X["fuel_type"] = X["fuel_type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
X["seller_type"] = X["seller_type"].map({"Dealer": 0, "Individual": 1, "Trustmark Dealer": 2})
X["transmission_type"] = X["transmission_type"].map({"Manual": 0, "Automatic": 1})

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 5. Save the trained model as a .pkl file
with open("rf_regressor.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training and saving completed!")
