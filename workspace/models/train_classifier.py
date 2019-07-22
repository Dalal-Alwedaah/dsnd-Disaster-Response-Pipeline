import sys
from sqlalchemy import create_engine

import pandas as pd 
import numpy as np
import pickle
import nltk

nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier

def load_data(database_filepath):
    ''' oad data from database '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    category_names = Y.columns
    return X , Y , category_names


def tokenize(text):
    ''' process text data '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' build a machine learning model '''
    pipeline  = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=0))))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2))
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    ''' show the accuracy, precision, and recall of the tuned model'''
    Y_pred = model.predict(X_test)
    for i in range(36):
        print("classification report for " + Y_test.columns[i]
              ,'\n', classification_report(Y_test.values[:,i],Y_pred[:,i])
              ,'\n accuracy:', accuracy_score(Y_test.values[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    ''' export the model as a pickle file '''
    pickle.dump(model, open(model_filepath, 'wb'))   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()