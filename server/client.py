import requests
import json
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print('Client start...')

header = {'Content-Type': 'application/json', 'Accept': 'application/json'}

# Save test data
with open('data/text_test.pkl', 'rb') as file:
    text = pickle.load(file)
with open('data/labels_test.pkl', 'rb') as file:
    labels = pickle.load(file)

print('Request...')

resp = requests.post('http://127.0.0.1:5000/predict',
                     data=json.dumps(text),
                     headers=header)

prediction = json.loads(resp.json())
prediction = [prediction[key] for key in prediction]

# print(f"Prediction: {prediction}")

y_pred = list()

for i in prediction:
    if i == 'About':
        y_pred.append(1)
    else:
        y_pred.append(0)

# Print scores
acc_score = accuracy_score(labels, y_pred)
pr_score, re_score, f_score, _ = precision_recall_fscore_support(labels,
                                                                 y_pred,
                                                                 average='binary')
print(f"Accuracy score: {round(acc_score, 3)}")
print(f"Precision score: {round(pr_score, 3)}")
print(f"Recall score: {round(re_score, 3)}")
print(f"F-score: {round(f_score, 3)}") # F-score: 0.602

