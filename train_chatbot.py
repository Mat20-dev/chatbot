#Importe la bibliothèque NLTK
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#Importe la bibliothèque JSON
import json

#Importe la bibliothèque PICKLE
import pickle

#Importe la bibliothèque NUMPY
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#Importe la bibliothèque RANDOM
import random

# Liste pour stocker tous les mots trouvés
words = []

#Liste pour stocker les différentes classes
classes = []

#Liste pour stocker la liste des documents
documents = []

#Liste des caractères à ignorer
ignore_words = ['?', '!']

#Lecture du fichier intents.json
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:

    #Tokenise chaque mot (Découpage des mots dans la phrase)
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        #Ajoute la liste des mots et son tag associé dans la liste des documents
        documents.append((w, intent['tag']))

        #Ajoute le tag à la liste des classes s'il n'est pas présent
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Met chaque mot en minuscules et supprimer les doublons
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#Trie la liste des classes et supprime les doublons
classes = sorted(list(set(classes)))

#Affiche le nombre de combinaisons entre patterns et intents qui ont été ajoutés
print(len(documents), "documents")

#Affiche le nombre de classes (intents)
print(len(classes), "classes", classes)

#Affiche le nombre total de mots uniques (lemmatisés)
print(len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes,open('classes.pkl', 'wb'))

#Crée une liste vide qui contiendra un ensemble de données d'apprentissage
training = []

#Crée un tableau vide pour notre sortie
output_empty = [0] * len(classes)

#Parcourt l'ensemble des données d'apprentissage et génère la liste de mots pour chaque phrase
for doc in documents:

    #Initialise notre ensemble de mots
    bag = []

    # Liste des mots segmentés correspondant au pattern
    pattern_words = doc[0]

    #lemmatise chaque mot et crée le mot de base afin de mettre en évidence les mots associés
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    #Crée le tableau de mots : ajouter 1 si le mot apparaît dans la phrase, sinon 0
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    #La sortie est 0 pour chaque tag et 1 pour le tag actuelle (pour chaque pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Mélange aléatoirement les données d'apprentissage et les convertir en tableau NumPy
random.shuffle(training)

#Crée les listes d'entraînement et de test : X pour les patterns, Y pour les intents
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

print("Training data created")

#Crée un modèle à 3 couches. Première couche : 128 neurones, deuxième couche : 64 neurones et une troisième couche de sortie contenant un nombre de neurones
#Égal au nombre d'intentions afin de prédire l'intention de sortie à l'aide de la fonction softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile le modèle : l'optimiseur SGD avec Nesterov donne de bons résultats
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Entraîne le modèle puis l'enregistre
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")