from skills import MLSkills

import re
import json
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Класс-обработчик возможных событий
class RecognitionHandler():

    def __init__(self):
        
        #Инициализция dataset
        with open("dataset.json", "r", encoding='utf-8') as file:
            self.capitals = json.load(file)

            self.INTENTS = self.capitals["intents"]

        #Инициализация ML-инструментов
        self.vectorizer = CountVectorizer() #настройки ngram_range, analyzer
        self.model = RandomForestClassifier()  #настройки n_estimators, max_depth
        self.classifier_probability = LogisticRegression()

        #Инициализация класса функций
        self.skill = MLSkills()

    def filter_text(self, text):
        text = text.lower()
        text = text.strip()
        pattern = r"[^\w\s]"
        text = re.sub(pattern, "", text)
        return text

    def counter_words(self, text):
        self.handle_text = self.filter_text(text=text)

        self.count = 0
        self.count = len(self.handle_text.split())
        if self.count == 1:
            #Вызов функции для обработки 1 слова
            self.one_word_handler(self.handle_text)
        else:
            #Вызов функции для обработки текста из множества слов
            self.train_model(self.handle_text)

    #Создание всех возможных векторов
    def create_vectors(self):
        self.X = []
        self.y = []
        for intent in self.INTENTS:
            examples = self.INTENTS[intent]["examples"]
            for example in examples:
                example = self.filter_text(example)
                if len(example) < 2:
                    continue
                self.X.append(example)
                self.y.append(intent)

    def train_vectorizer(self):
        self.create_vectors()
        self.vectorizer.fit(self.X)
        self.vecX = self.vectorizer.transform(self.X)

    def train_model(self, text):
        #Вызов всех необходимых функций
        self.train_vectorizer()

        #Обучение модели
        self.model.fit(self.vecX, self.y)
        self.classifier_probability.fit(self.vecX, self.y)

        vec_text = self.vectorizer.transform([text])
        final_intent_ml = self.model.predict(vec_text)[0]

        index_of_best_intent = list(self.classifier_probability.classes_).index(final_intent_ml)
        probabilities = self.classifier_probability.predict_proba(self.vectorizer.transform([text]))[0]

        best_intent_probability = probabilities[index_of_best_intent]

        #Проверяем уверенность интеллекта в своем решении (в зависимости от datasets менять показатель)
        if best_intent_probability > 0.225:
            print(f"Я уверен в этом намерении {final_intent_ml} на {best_intent_probability}")
            answer = getattr(self.skill, final_intent_ml)
            answer()        
        else:
            print("Фраза была не распознана")
            self.skill.recognition_error()


    def one_word_handler(self, text):
        final_intent = ""

        #Перебираем до тех пор пока не найдем совпадение больше чем 0.75
        for intent in self.INTENTS:
            examples = self.INTENTS[intent]["examples"]
            for example in examples:
                diffrence = SequenceMatcher(None, text, example).ratio()
                #Сравниваем входное слово с словом из json-объекта
                if diffrence > 0.75:
                    final_intent = intent
                    break
                else:
                    continue

        if final_intent:
            print(f"Я уверен в этом намерении {final_intent} больше чем 75%")
        else:
            print("Фраза была не распознана")

if __name__ == "__main__":
    while True:
        text = input("Введите текст вашего намерения: ")
        RecognitionHandler().counter_words(text=text)
