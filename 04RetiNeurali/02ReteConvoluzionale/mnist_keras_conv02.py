from random import randint

import matplotlib.pyplot as plt
import numpy
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.utils import np_utils

# ottengo il dataset MNIST già suddiviso in dataset X e Y, di addestramento e di test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# visualizzo 4 cifre random
for i in range(4):
    plt.subplot(2, 2, (i + 1))
    # in 'shape[0]' è contenuto il numero di esempi del dataset (in questo caso di addestramento)
    plt.imshow(X_train[randint(0, X_train.shape[0])], cmap=plt.get_cmap('gray'))
plt.show()

# imposto un seed random in modo da ottenere risultati replicabili, d'ora in avanti
numpy.random.seed(1234)

# modifico le matrici di pixel in modo da ottenere una matrice di pixels monocromatici
# usando Tensorflow, il canale del colore è l'ultimo dopo le dimensioni (ncifre, dimy, dimx, ncanalicolore)
# con altri backend (ad esempio Theano) il canale va prima delle dimensioni (ncifre, ncanalicolore, dimy, dimx)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalizzo i valori dei pixel portandoli dal range intero 0-255 al range in virgola mobile 0.0-1.0
# visto che sono array numpy, è sufficiente eseguire l'operazione direttamente sull'array
X_train = X_train / 255
X_test = X_test / 255

# modifico gli array dei risultati ("ground truth") in modo siano in formato 'one hot encode'
# quindi i valori interi corrispondenti alla classe della cifra (0, 1, 2, ..., 9) vengono
# codificati in stringhe posizionali di 0 ed 1
# esempi:
#   0 --> 1,0,0,0,0,0,0,0,0,0
#   1 --> 0,1,0,0,0,0,0,0,0,0
#   2 --> 0,0,1,0,0,0,0,0,0,0
#  ....
#   9 --> 0,0,0,0,0,0,0,0,0,1
# in questo modo è più semplice ottenere un risultato significativo dalla rete neurale, in quanto
# ogni cifra posizionale corrisponderà ad un neurone dello strato di output che si attiverà o meno
# a seconda del risultato della classificazione della rete neurale
# in 'shape[0]' continuerà ad essere contenuto il numero di cifre del dataset
# in 'shape[1]' ci sarà invece il numero di cifre posizionali, corrispondente al numero di classi possibili
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classi = y_train.shape[1]

# definisco un modello di rete neurale convoluzionale
# come dimensione del layer di ingresso uso quelle del train set (esclusa la dimensione iniziale del numero di cifre del dataset)
input_layer = Input(shape=X_train.shape[1:], name="input_layer")
inner_layer = Conv2D(30, (5, 5), activation="relu", name="conv_layer_1")(input_layer)
inner_layer = MaxPooling2D(pool_size=(2, 2), name="maxpool_layer_1")(inner_layer)
inner_layer = Conv2D(15, (3, 3), activation="relu", name="conv_layer_2")(inner_layer)
inner_layer = MaxPooling2D(pool_size=(2, 2), name="maxpool_layer_2")(inner_layer)
inner_layer = Dropout(rate=0.2, name="drop_layer")(inner_layer)
inner_layer = Flatten(name="flatten_layer")(inner_layer)
inner_layer = Dense(128, activation="relu", name="dense_layer_1")(inner_layer)
inner_layer = Dense(50, activation="relu", name="dense_layer_2")(inner_layer)
output_layer = Dense(num_classi, activation="softmax", name="output_layer")(inner_layer)

model = Model(inputs=[input_layer], outputs=[output_layer])
model.summary()

# compilo il modello indicando che tipo di loss_function devo utilizzare,
# il tipo di ottimizzatore e le metriche che voglio vengano calcolate
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# addestro il modello
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=10, batch_size=256, verbose=1)

# valuto il modello
valutazioni = model.evaluate(X_test, y_test, verbose=1)
print("Errore del modello: {:.2f}%".format(100 - valutazioni[1] * 100))
