import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle
##Reading the data from csv file
data =pd.read_csv("weather_classification_data.csv")
# printing the first few rows  and columns of the data
print(data.head())
print(data.columns)
#data visualizing using pairplot
plt.figure(figsize=(10,8))
sns.pairplot(data[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Weather Type']], hue='Weather Type')
plt.title("Pair Plot of Weather Features by Weather Type")
plt.show()
#encoding categorical data to numerical data
label_encoder_cloud = LabelEncoder()
label_encoder_location = LabelEncoder()
#fiting and transforming the categorical data to numerical data using label encoder
data['Cloud Cover Encoded'] = label_encoder_cloud.fit_transform(data['Cloud Cover'])
data['Location Encoded'] = label_encoder_location.fit_transform(data['Location'])

# Defining the input features (X) and target label (y)
X = data.drop(['Weather Type', 'Cloud Cover', 'Location', 'Season'], axis=1)  # Droping unnecessary columns
y = data['Weather Type']# to predict
#splititng the data into training and testing data  using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#selecting our model to be a random forest classifier and fitting our x and y values to our model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
#predicting the y values using the  test dataa by our model
y_pred = model.predict(X_test)
print(y_pred)
#calculating the accuracy of our model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)
#saving our model using joblib


with open('weather_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)
feature_names = X.columns
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

with open('label_encoder_cloud.pkl', 'wb') as f:
    pickle.dump(label_encoder_cloud, f)

with open('label_encoder_location.pkl', 'wb') as f:
    pickle.dump(label_encoder_location, f)
