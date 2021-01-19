import argparse
import os

import tensorflow as tf
from costanti import *
from generatore_dati import Generatore
from modello import get_input_output_layer
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)


def ctc_lambda(args_list):
    """
    Funzione lambda per gestire il costo CTC nella funzione LOSS
    :param y_pred: valori predetti
    :param y_true: valori attesi
    :param input_length: lunghezza dei dati predetti
    :param label_length: lunghezza dei dati attesi
    :return: costo CTC
    """
    y_pred, y_true, input_length, label_length = args_list
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)


# preparo gli argomenti da linea di comando
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", required=False, type=str, default=DEFAULT_DATASET_DIR)
parser.add_argument("--string-len", required=False, type=int, default=DEFAULT_STRING_LEN)
parser.add_argument("--weights-dir", required=False, type=str, default=DEFAULT_WEIGHTS_DIR)
args = parser.parse_args()

print("Addestramento del modello")
print("-" * 80)
print("Directory dataset: %s" % args.dataset_dir)
print("Directory pesi: %s" % args.weights_dir)
print("Lunghezza stringa: %d" % args.string_len)
print("-" * 80)
print()

os.makedirs(args.weights_dir, exist_ok=True)

# preparo i due generatori di dati (di training e di validation)
train_gen = Generatore(dataset_dir=os.path.join(args.dataset_dir, DATASET_TRAIN_DIR))
val_gen = Generatore(dataset_dir=os.path.join(args.dataset_dir, DATASET_VAL_DIR))

# preparo il modello
input_data, y_pred = get_input_output_layer(img_w=train_gen.image_size[1],
                                            img_h=train_gen.image_size[0],
                                            alfabeto_len=len(ALFABETO))

train_gen.set_output_shape(y_pred.shape)
val_gen.set_output_shape(y_pred.shape)

# visto che sono in addestramento, preparo una funzione di loss specifica
# che possa utilizzare il costo CTC
y_true = Input(shape=[args.string_len], dtype="float32", name="y_true")
y_pred_length = Input(shape=[1], dtype="int64", name="input_length")
y_true_length = Input(shape=[1], dtype="int64", name="y_true_length")
loss_layer = Lambda(ctc_lambda, output_shape=(1,), name="ctc")([y_pred, y_true, y_pred_length, y_true_length])

# completo il modello utilizzando il layer di input, quello di output e la funzione di loss
model = Model(inputs=[input_data, y_true, y_pred_length, y_true_length], outputs=loss_layer)
model.summary()

# compilo il modello aggiungendo anche l'ottimizzatore
model.compile(loss={"ctc": lambda y_t, y_p: y_p},
              optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5))

# preparo una callback per memorizzare i pesi migliori riscontrati durante l'addestramento
chkpnt_callback = ModelCheckpoint(filepath=os.path.join(args.weights_dir, BEST_WEIGHTS_FILE),
                                  save_weights_only=True,
                                  save_best_only=True,
                                  verbose=1,
                                  monitor="val_loss",
                                  mode="min")

# in caso il file dei pesi esista, lo legge in modo da ripartire con l'addestramento
if os.path.exists(os.path.join(args.weights_dir, BEST_WEIGHTS_FILE)):
    model.load_weights(os.path.join(args.weights_dir, BEST_WEIGHTS_FILE))
    print("Trovati pesi, eseguo un addestramento ulteriore")
else:
    print("Nessun peso trovato, eseguo un addestramento da zero")

# addestro il modello
model.fit(x=train_gen,
          epochs=EPOCHS,
          callbacks=[chkpnt_callback],
          validation_data=val_gen,
          shuffle=False,
          steps_per_epoch=len(train_gen),
          validation_steps=len(val_gen),
          verbose=1)

print("Addestramento completato!")
