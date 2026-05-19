#Importe la bibliothèque NLTK
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#Importe la bibliothèque PICKLE
import pickle

#Importe la bibliothèque NUMPY
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')

#Importe la bibliothèque JSON
import json

#Importe la bibliothèque RANDOM
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    #Découpe les mots en tableau (tokenisation)
    sentence_words = nltk.word_tokenize(sentence)

    #Crée une forme abrégée pour chaque mot (lemmatisation)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    #Retourne le tableau de mots : 0 ou 1 pour chaque mot présent dans la phrase
    return sentence_words

def bow(sentence, words, show_details=True):
    #Tokenise le pattern (nettoye et découpe la phrase)
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    #Matrice de N mots : matrice lexicale représentant tout le vocabulaire
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                #Attribue la valeur 1 si le mot actuel correspond à la position indiquée dans le vocabulaire
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    #Filtre les prédictions en dessous d'un certain seuil
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    #Trie par ordre décroissant de probabilité
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    #Récupére une réponse correspondant à l’intention la plus probable et choisit une réponse au hasard
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    #Génére la réponse du chatbot à partir du texte de l'utilisateur
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

#Création de l'interface graphique avec Tkinter
import tkinter
from tkinter import *

def send():
    # Envoye le message de l'utilisateur et affiche la réponse du chatbot
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You : " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot : " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

#Configuration de la fenêtre principale
base = Tk()
base.title("ChatBot - Posez-moi une question !")
base.geometry("1200x1000")
base.resizable(width=FALSE, height=FALSE)

#Crée la fenêtre de discussion et affiche un message d'accueil
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.insert(END, "Bienvenue dans le chatbot ! Tapez votre message pour commencer.\n\n")
ChatLog.config(state=DISABLED)

#Associe la barre de défilement à la fenêtre de discussion
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Crée un bouton pour envoyer un message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Envoyer", width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command= send)

#Crée la zone de saisie du message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

#Place tous les éléments à l'écran
scrollbar.place(x=1176, y=6, height=886)
ChatLog.place(x=6, y=6, height=886, width=1170)
EntryBox.place(x=6, y=901, height=90, width=900)
SendButton.place(x=911, y=901, height=90, width=283)

base.mainloop()