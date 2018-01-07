from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, TimeDistributed
from keras.models import model_from_json
from keras import metrics
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
import itertools

num_nb_epoch = 2
maxlen = 15  # cut texts after this number of words (among top max_features most common words)
batch_size = 20

#Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cv = pickle.load(open("vocab.p", "rb"))
max_features = len(cv.vocabulary_)
print('The vocabulary size is {0}'.format(max_features))

print('Loading data...')
X = np.loadtxt("sequence_data")
y = np.loadtxt("sequence_labels")
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 256, input_length=maxlen))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dropout(0.6)) # try different dropout rates- 0.5
model.add(Dense(8))
model.add(Dense(3, activation='softmax'))

# Use either adam, Nadam, Adagrad,or RMSprop as optimizer.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=num_nb_epoch,
          validation_data=[X_test, y_test])

print('Saving the model to \"model.json\"')

#Save the model to json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
print('Saving the weights to model.h5')
model.save_weights("model.h5")

print('Creating Confusion Matrix Based on Test Data...')
y_pred = model.predict(x=X_test, verbose=1)
y_pred_new = np.empty(len(y_pred))
y_test_new = np.empty(len(y_test))
class_names = ('Positive', 'Neutral', 'Negative')

for row_num in range(0, len(y_pred)):
    y_pred_new[row_num] = np.argmax(y_pred[row_num])
    y_test_new[row_num] = np.argmax(y_test[row_num])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true=y_test_new, y_pred=y_pred_new)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.figure()
class_totals = np.sum(y, axis=0) / len(y)
y_pos = np.arange(len(class_names))

plt.bar(y_pos, class_totals, align='center', alpha=0.5)
plt.xticks(y_pos, class_names)
plt.ylabel('Frequency')
plt.title('Frequency of Sentiments in Data Set')

plt.show()
