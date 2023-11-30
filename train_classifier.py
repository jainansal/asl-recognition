import pickle
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
data_dict = pickle.load(open('data.pkl', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

mini = 100000
for ddata in data:
    mini = min(mini, len(ddata))

cnt = 0
i = 0
while i < len(data):
    if len(data[i]) == mini:
        i += 1
    else:
        data.pop(i)
        labels.pop(i)
        cnt += 1

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('Accuracy:', score)

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
