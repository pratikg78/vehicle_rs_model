import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import hstack
import joblib
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

data_path = os.path.join(DATA_DIR, 'Updated_Car_Sales_Data.xlsx')
df = pd.read_excel(data_path)

# Drop rows with missing critical values
df = df.dropna(subset=['Car Make', 'Car Model', 'Year', 'Price', 'Fuel Type', 'Transmission']).reset_index(drop=True)

# Features
categorical_features = ['Car Make', 'Fuel Type', 'Transmission']
numerical_features = ['Year', 'Price']

# One-hot encode categorical features
encoder = OneHotEncoder()
encoded_cat = encoder.fit_transform(df[categorical_features])

# Scale numerical features
scaler = StandardScaler()
scaled_num = scaler.fit_transform(df[numerical_features])

# Combine features  
features_combined = hstack([encoded_cat, scaled_num])

# Dimensionality reduction with PCA
n_components = min(30, features_combined.shape[1])
pca = PCA(n_components=n_components)
reduced_features = pca.fit_transform(features_combined.toarray())

# Build Nearest Neighbors model with cosine similarity
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(reduced_features)

# def encodeCatInp(CarMake,FuelType, Transmission ):
#     return [encoder.fit_transform([str(CarMake)]),encoder.fit_transform([str(FuelType)]),encoder.fit_transform([str(Transmission)])]


# def encodeNumInpt(Year, Price):
#     return[scaler.fit_transform(np.array(Year).reshape(1, -1)),scaler.fit_transform(np.array(Price).reshape(1, -1))]



# print(recommendCar(["Toyota","Gasoline","Manual",2016, 20000]))

joblib.dump(nn_model, 'models/veh_rec_model.pkl')
joblib.dump(encoder, "models/cat_encoder.pkl")
joblib.dump(scaler, "models/num_scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(df, "models/dataFrame.pkl")


# # Choose an example car
# index = 207
# distances, indices = nn_model.kneighbors([reduced_features[index]], n_neighbors=6)

# # Show recommendations
# recommended_indices = indices[0][1:]
# recommendations = df.iloc[recommended_indices][['Car Make', 'Car Model', 'Year', 'Price']]
# print(f"Recommendations for car at index {index} ({df.loc[index, 'Car Make']} {df.loc[index, 'Car Model']}):")
# print(recommendations)

# # Evaluate cosine similarity
# avg_cosine_similarity = 1 - distances[0][1:].mean()
# print(f"\nAverage Cosine Similarity of Recommendations: {avg_cosine_similarity:.4f}")




# joblib.dump(nn_model, 'models/_model.pkl')




# # Save encoders to disk
# joblib.dump(district_encoder, 'models/District_encoder.pkl')
# joblib.dump(village_encoder, 'models/Village_encoder.pkl')
# joblib.dump(soilQuality_encoder, 'models/Soil_Quality_encoder.pkl')





