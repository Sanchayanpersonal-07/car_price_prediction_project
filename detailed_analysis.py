import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load Dataset
df = pd.read_csv('cardekho_dataset.csv')

# Dataset summary
print("Dataset Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Dataset Information
print("\nDataset Information:")
df.info()

print("Shape of the dataset:",df.shape) 

# Car with max and min selling price
max_selling_car = df.loc[df['selling_price'].idxmax(), 'car_name']
min_selling_car = df.loc[df['selling_price'].idxmin(), 'car_name']
max_selling_price = df['selling_price'].max()
min_selling_price = df['selling_price'].min()

print(f"Maximum selling price: {max_selling_price} (Car: {max_selling_car})")
print(f"Minimum selling price: {min_selling_price} (Car: {min_selling_car})")

# Pie Chart of Fuel Types
fuel_type_counts = df['fuel_type'].value_counts()
colors = sns.color_palette('pastel')[0:len(fuel_type_counts)]
plt.figure(figsize=(8, 6))
plt.pie(fuel_type_counts, labels=fuel_type_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.axis('equal')
plt.title('Distribution of Cars based on Fuel Type')
plt.tight_layout()
plt.show()

# Distribution of Seller Types
selling_type_counts = df['seller_type'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(selling_type_counts.index, selling_type_counts.values, color=['blue', 'brown'])
plt.title('Distribution of Selling Types')
plt.xlabel('Type of Seller')
plt.ylabel('Number of Cars Sold')
plt.tight_layout()
plt.show()

# Plotting the transmission types as a pie chart
transmission_counts = df['transmission_type'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Transmission Types')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Top 10 Car Brands Distribution
brand_distribution = df['brand'].value_counts().head(10)
plt.figure(figsize=(8, 8))
plt.pie(brand_distribution, labels=brand_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Vehicle Brands Distribution')
plt.axis('equal')  
plt.tight_layout()
plt.show()

# Year vs Selling Price
df['year'] = 2024 - df['vehicle_age']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='selling_price', data=df, color='purple')
plt.title('Year vs Selling Price')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.tight_layout()
plt.show()

# Selling Price Distribution 
plt.figure(figsize=(10, 6)) 
sns.histplot(df['selling_price'], bins=30, kde=True, color='green') 
plt.title('Distribution of Selling Prices') 
plt.xlabel('Selling Price') 
plt.ylabel('Frequency') 
plt.tight_layout() 
plt.show() 

# Mileage vs Selling Price 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='mileage', y='selling_price', data=df, color='blue') 
plt.title('Mileage vs Selling Price') 
plt.xlabel('Mileage (kmpl)') 
plt.ylabel('Selling Price') 
plt.tight_layout() 
plt.show() 

# Engine vs Selling Price 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='engine', y='selling_price', data=df, color='red') 
plt.title('Engine vs Selling Price') 
plt.xlabel('Engine (cc)') 
plt.ylabel('Selling Price') 
plt.tight_layout() 
plt.show() 

# Vehicle Age Distribution 
plt.figure(figsize=(10, 6)) 
sns.histplot(df['vehicle_age'], bins=30, kde=True, color='orange') 
plt.title('Distribution of Vehicle Ages') 
plt.xlabel('Vehicle Age (years)') 
plt.ylabel('Frequency') 
plt.tight_layout() 
plt.show() 

# Count of Cars by Brand 
plt.figure(figsize=(10, 6)) 
sns.countplot(y='brand', data=df, order=df['brand'].value_counts().index, palette='viridis') 
plt.title('Count of Cars by Brand') 
plt.xlabel('Count') 
plt.ylabel('Brand') 
plt.tight_layout() 
plt.show()

# Correlation Heatmap
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Encoding categorical variables using LabelEncoder
le_fuel = LabelEncoder()
df['fuel_type'] = le_fuel.fit_transform(df['fuel_type'])

le_seller = LabelEncoder()
df['seller_type'] = le_seller.fit_transform(df['seller_type'])

le_transmission = LabelEncoder()
df['transmission_type'] = le_transmission.fit_transform(df['transmission_type'])

# Select relevant features and target variable
X = df[['vehicle_age', 'km_driven', 'fuel_type', 'seller_type', 'transmission_type', 'mileage', 'max_power', 'engine', 'seats']]
y = df['selling_price']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions and evaluate the Linear Regression model
y_pred_lr = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Linear Regression R^2: {lr_r2}")

# 2. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rf_rmse}")
print(f"Random Forest R^2: {rf_r2}")

# Compare the two models and return the best one based on R^2
if lr_r2 > rf_r2:
    print("Linear Regression is the best model with R^2:", lr_r2)
    best_model = lr_model
else:
    print("Random Forest is the best model with R^2:", rf_r2)
    best_model = rf_model
