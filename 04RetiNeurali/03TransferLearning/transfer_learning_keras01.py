# uso la libreria per scaricare automaticamente un certo numero di immagini di cinciarelle e corvi
# NOTA: il dataset scaricato può contenere immagini non adatte all'addestramento, quindi va verificato
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
arguments = {"keywords": "cinciarella", "limit": 100, "print_urls": False, "format": "jpg", "size": ">400*300"}
response.download(arguments)
arguments = {"keywords": "corvo", "limit": 100, "print_urls": False, "format": "jpg", "size": ">400*300"}
response.download(arguments)

# Definisco un modello 'backbone' utilizzando il classificatore 'MobileNet'
# pre-addestrato con dataset 'ImageNet'
# Indico che non voglio i top layer del classificatore, perchè userò i miei
# Indico anche che l'ultimo layer sarà un GlobalAveragePool, che consente di
# limitare ulteriormente i parametri da gestire
from keras.applications import MobileNetV2, mobilenet_v2
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

backbone = MobileNetV2(weights='imagenet', include_top=False, pooling="avg")

# indico i layer del backbone come "non addestrabili", in modo da non modificarli
for l in backbone.layers:
    l.trainable = False

# definisco gli ultimi layer di classificazione, usando come ingresso le uscite del backbone
x = backbone.output
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

# il mio modello complessivo avrà gli stessi ingressi del backbone e l'uscita che ho definito
# nei mei top-layers
model = Model(inputs=backbone.input, outputs=preds)

# visualizzo un sommario del modello complessivo
model.summary()

# compilo il modello con una loss di classificazione, l'ottimizzatore Adam ed aggiungendo l'accuracy come metrica
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])

# creo un generatore di immagini che utilizzi la funzione di preprocessing necessaria al modello MobileNetV2
train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input)

# indico al generatore di immagini dove si trovano le immagini, le dimensioni da usare, il formato colore da usare,
# il batch_size con cui costruire i vari batch, il tipo di classificazione, e se deve mischiare il dataset
train_generator = train_datagen.flow_from_directory('./downloads',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

# addestro il modello usando il generatore di immagini definito in precedenza, indicando
# quanti cicli eseguire per ogni epoca (lo calcolo dividendo l'ampiezza del dataset per il batch_size)
# ed utilizzando 10 epoche in tutto
from math import ceil

model.fit_generator(generator=train_generator,
                    steps_per_epoch=ceil(train_generator.n / train_generator.batch_size),
                    epochs=10,
                    verbose=1)

import urllib
import numpy as np


def download_image(url, filename):
    # eseguo una GET sulla url passata come parametro
    with urllib.request.urlopen(url) as url_get:
        # apro il file su cui scriverò l'immagine
        with open(filename, "wb") as f:
            # leggo dalla GET e scrivo sul file
            f.write(url_get.read())


def load_image(img_path):
    # carico l'immagine dal file
    img = image.load_img(img_path, target_size=(224, 224))
    # trasformo l'immagine in un array Numpy
    # lo shape dell'array sono (altezza, larghezza, canali colore)
    # quindi in questo caso (224, 224, 3)
    img_array = image.img_to_array(img)
    # aggiungo una dimensione all'inizio
    # lo shape diventa (1, 224, 224, 3)
    # dove "1" indica quante immagini sono presenti nel batch
    img_array_batch = np.expand_dims(img_array, axis=0)
    # normalizzo i valori da 0..255 a 0..1
    img_array_batch /= 255.

    return img_array_batch


# verifico la predizione su un disegno di ciciarella (immagine non utilizzato in training)
download_image("http://www.connemara.it/natura/fauna/uccelli/cinciarella%20foto/cinciarella%20disegno.jpg",
               "tempcinciarella.jpg")

img_di_test = load_image("tempcinciarella.jpg")
predizione = model.predict(img_di_test)
print("Predizione cinciarella:", predizione)

# analogamente per una immagine di corvo
download_image("http://3.bp.blogspot.com/-FV6e0kERgFE/VQMHK0m0E3I/AAAAAAAAFpA/7BeFPWSq8Tk/s1600/Nevermore.jpg",
               "tempcorvo.jpg")

img_di_test = load_image("tempcorvo.jpg")
predizione = model.predict(img_di_test)
print("Predizione corvo:", predizione)
