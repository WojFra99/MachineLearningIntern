import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import random



#Data set acquisition, splitting into training and test sets, and standardization.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
df = pd.read_csv(url, compression='gzip', header=None)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
y_train_b = keras.utils.to_categorical(y_train - 1, 7)
y_test_b = keras.utils.to_categorical(y_test - 1, 7)



#Function for displaying confusion matrix.
def confusion_matrix_plot(y_pred, model):
  cm = confusion_matrix(y_test, y_pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.title(f'Confusion matrix ({model})')
  plt.xlabel('True label')
  plt.ylabel('Predicted label')
  plt.show()



#Simple heuristic model that randomly assigns class labels
def heuristic_model():
  N = y_test.size
  n = y.size
  y_pred=y[[random.randint(0, n-1) for i in range(N)]]
  print('Accuracy: %.3f' % accuracy_score(y_pred, y_test))
  print('F1: %.3f' % f1_score(y_test, y_pred, average='weighted'))
  confusion_matrix_plot(y_pred, "heuristic model")



#Decision Tree and Random Forest models from Scikit-learn library
def decision_tree_and_random_forest_model():
  dt = DecisionTreeClassifier()
  dt.fit(X_train_std, y_train)
  y_pred=dt.predict(X_test_std)
  print('Accuracy: %.3f' % accuracy_score(y_pred, y_test))
  print('F1: %.3f' % f1_score(y_test, y_pred, average='weighted'))
  confusion_matrix_plot(y_pred, "Decision Tree")
  print("=========================================================================")


  rf = RandomForestClassifier(n_estimators=150, random_state=123, max_depth=3)
  rf.fit(X_train, y_train)
  y_pred=rf.predict(X_test_std)
  print('Accuracy: %.3f' % accuracy_score(y_pred, y_test))
  print('F1: %.3f' % f1_score(y_test, y_pred, average='weighted'))
  confusion_matrix_plot(y_pred, "Random Forest")



#Neural network model
def NN_model():
  model = Sequential()
  model.add(Dense(units=256, activation='relu', input_shape=(54,)))
  model.add(Dense(units = 256, activation='relu'))
  model.add(Dense(units = 128, activation='relu'))
  model.add(Dense(units = 7, activation='sigmoid'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(X_train_std, y_train_b, epochs=10, verbose=1, validation_data=(X_test_std, y_test_b))

  plt.figure(figsize=(8, 4))
  plt.plot(history.history['accuracy'], label='training accuracy')
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_accuracy'], label='validation accuracy')
  plt.plot(history.history['val_loss'], label='validation loss')
  plt.title('Training Curves')
  plt.xlabel('Epoch')
  plt.xticks(range(0, 10, 1))
  plt.legend()
  plt.show()



#Function which find a good set of hyperparameters for the NN
def find_set_of_hp():
  def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(54,)))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.1)))
    model.add(Dense(units = 7, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  hyperparameters = HyperParameters()
  hyperparameters.Int('units', min_value=32, max_value=512, step=32)
  hyperparameters.Float('dropout', min_value=0.0, max_value=0.2, step=0.1)

  tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5)
  tuner.search(X_train_std, y_train_b, epochs=5, validation_data=(X_test_std, y_test_b))

  best_model = tuner.get_best_models(num_models=1)[0]
  return best_model.summary()



while True:
    model = input("""Choose a model:
1. Heuristic
2. Decision Tree, Random Forest
3. Neural Network 
4. NN model with good set of hyperparameters.
5. Exit
""")

    if model == "5":
        break

    if model == "1":
        heuristic_model()
        continue
    elif model == "2":
        decision_tree_and_random_forest_model()
        continue
    elif model == "3":
        NN_model()
        continue
    elif model == "4":
        find_set_of_hp()
        continue
    else:
        print("There is no such model. Select a number from 1 to 4.")