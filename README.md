# DS.Test_task1
### Task: <br>
learning classifier that will determine whether or not the text contains descriptive content ("About or "None" label).

### Files: <br>
* **research1.ipynb** - Classic research of data and models. Data scaling using RandomUnderSampler, RandomOverSampler, SMOTE. Vectorizing of text using LabelEncoder, CountVectorizer, TfidfVectorizer. Construction of LinearRegression and NN models.
* **research2. NLP.ipynb** - Preprocesing and training spaCy models.
* **base_config.cfg, config.cfg** - Configuration files for training spaCy models.
* **research3. Ensembles.ipynb** - Construction of ensemble models (bagging, RandomForestClassifier, StackingClassifier, XGBoost).
* **result_functions.py** - Contains 2 functions (print_scores, plot_history_nn) for calculating, printing and drawing score graphs.

* ./server/**server.py** - Server (Flask) for get post requests ("http://127.0.0.1:5000//predict") and predict labels for text. Use Embedding NN model with the highest score (from "***research3. Ensembles.ipynb***"). Can get a string or a list of strings.
* ./server/**client.py** - A simple example with a client that takes test data, do request to server, receive a predictions and evaluating them. <br>

### Folders:<br>
* ./server/ - Server, client, models files for prediction.

* ./data/ - Folder with data.
* ./models/ - Saved models.
* ./output/ - Saved spaCy models.

### How to predict labels for text
1. Run "server.py"
2. Do POST request to "http://127.0.0.1:5000//predict". Example:

```
import requests
import json

header = {'Content-Type': 'application/json', 'Accept': 'application/json'}
resp = requests.post('http://127.0.0.1:5000/predict',
                     data=json.dumps(list_text), # String or list of string
                     headers=header)
                     
prediction = json.loads(resp.json())
prediction = [prediction[key] for key in prediction] # Return a list of labels ('About' or 'None')
```
