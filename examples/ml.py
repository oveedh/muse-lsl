import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

data = pd.read_csv("/Users/oveedharwadkar/muse/muse-lsl/examples/data.csv")

#sort by image as image is the independent variable
data['Image'] = data['Image'].astype('category')
#take the mean of each Alpha and Beta of each independent variable
image_data = data.groupby("Image").mean().round(2)
#print it out
print(image_data)

def train_svmm_model():
    data = pd.read_csv("/Users/oveedharwadkar/muse/muse-lsl/examples/data.csv")

    # Assuming 'X' contains your features (Beta and Alpha frequencies) 
    # and 'y' contains the categories
    X = data[['Beta','Alpha']].values
    y = data['Image'].values

    # Splitting the data into training and validation sets (adjust test_size and random_state as needed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizing the features (scaling between 0 and 1)
    scaler = RobustScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)

    # Initializing SVM classifier (you can experiment with different kernels and parameters)
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')  # Example with RBF kernel

    # Training the SVM model
    svm_classifier.fit(X_train_normalized, y_train)

    # Making predictions on the validation set
    predictions = svm_classifier.predict(X_val_normalized)

    # Calculating accuracy
    accuracy = accuracy_score(y_val, predictions)
    #print("Accuracy:" + str(accuracy*100))

    return svm_classifier, scaler

def predict_with_svmm_model(svm_classifier, scaler, X_new):
    # Normalize the new data using the same scaler used for training/validation data
    X_new_normalized = scaler.transform(X_new)

    # Making predictions on the new data
    new_predictions = svm_classifier.predict(X_new_normalized)
    # Displaying the predicted labels
    #print("Predicted Labels for New Data:")
    #for prediction in new_predictions:
        #print(prediction)
    
    return new_predictions


#testing if code works:
#svm_classifier,scaler = train_svmm_model() #training the model 
#predict_with_svmm_model(svm_classifier, scaler, [[0.35, 0.1]])
#^^ testing the predictive model - replace 1, 1 with any {betaValue, alphaValue} you wish