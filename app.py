from flask import Flask, request,make_response
import joblib
import pandas as pd
import numpy as np
import os 
from flask_cors import CORS
from scipy.sparse import hstack

# from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

CORS(app)

encoder = joblib.load('models/cat_encoder.pkl')
nn_model = joblib.load('models/veh_rec_model.pkl')
scaler = joblib.load('models/num_scaler.pkl')
pca = joblib.load("models/pca.pkl")
df = joblib.load("models/dataFrame.pkl")

def encodeCatInp(CarMake, FuelType, Transmission):
    cat_input = pd.DataFrame([[CarMake, FuelType, Transmission]], columns=['Car Make', 'Fuel Type', 'Transmission'])
    return encoder.transform(cat_input)

def encodeNumInpt(Year, Price):
    num_input = np.array([[Year, Price]])  # shape (1, 2)
    return scaler.transform(num_input)


def recommendCar(inputFeatures):
    encoded_cat = encodeCatInp(inputFeatures[0], inputFeatures[1], inputFeatures[2])
    encoded_num = encodeNumInpt(inputFeatures[3], inputFeatures[4])

    combinedInput = hstack([encoded_cat, encoded_num])
    reduced_input = pca.transform(combinedInput.toarray())

    distances, indices = nn_model.kneighbors(reduced_input)
    recommendations = df.iloc[indices[0]]
    return recommendations[['Car Make', 'Car Model', 'Year', 'Price','Color']]



@app.route('/', methods=["GET"])
def home():
  return make_response("Backend for Vehicle Recommendation System",200)


@app.route('/predict', methods=['POST'])
def predict():
  try:

    data = request.get_json()
    
    if(not data):
      return make_response({
        "Error" : {
          "code": 400,
          "message": "Bad request"
        }
      }, 400)
    

    recommend_df = recommendCar([str(data["company"]),str(data["fuel"]), str(data["transmission"]), int(data["year"]),float(data["price"])])
    recommend_list = recommend_df.to_dict(orient='records')
    return make_response({
      "Success": {
      "code": 200,
      "prediction": recommend_list
    }}, 200)
  except Exception as e:
      return make_response({"Error": {"code": 500, "message": str(e)}}, 500)

if __name__ == '__main__':
    app.run(debug=True)
