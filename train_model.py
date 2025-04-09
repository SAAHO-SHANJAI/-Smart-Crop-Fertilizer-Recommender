# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load data
data = pd.read_csv('data/data_core.csv')

# Label Encoding
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

data['Soil Type'] = le_soil.fit_transform(data['Soil Type'])
data['Crop Type'] = le_crop.fit_transform(data['Crop Type'])
data['Fertilizer Name'] = le_fert.fit_transform(data['Fertilizer Name'])

# Features and Target
X = data[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name']

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Save Model and Encoders
os.makedirs('model', exist_ok=True)

with open('model/fertilizer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/soil_encoder.pkl', 'wb') as f:
    pickle.dump(le_soil, f)

with open('model/crop_encoder.pkl', 'wb') as f:
    pickle.dump(le_crop, f)

with open('model/fert_encoder.pkl', 'wb') as f:
    pickle.dump(le_fert, f)

print("✅ Model training complete and saved successfully.")
