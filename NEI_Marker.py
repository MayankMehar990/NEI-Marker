# Importing the Libraaries
import numpy as np
import pandas as pd
import sklearn
import sklearn_crfsuite
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_classification_report
# Importing libraries for metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
)
from sklearn.model_selection import KFold



# Loading the dataset
data = load_dataset("conll2003",trust_remote_code=True)

# Load label,and pos tag names for NER and POS tag
label_names = data['train'].features['ner_tags'].feature.names
pos_tag_names = data['train'].features['pos_tags'].feature.names
# Sentences is a processed dataset
sentences=[]
df=[]
for sentence in data['train']:
    # Get tokens,pos tags and NER tags for each sentence
    tokens = sentence['tokens']
    pos_tags = [pos_tag_names[tag] for tag in sentence['pos_tags']]
    ner_tags = [label_names[tag] for tag in sentence['ner_tags']]
    temp=[]
    for i,j,k in zip(tokens,pos_tags,ner_tags):
        temp.append((i,j,k))
    sentences.append(temp)



# Extracting the features    
def extract_features(sentence_with_tags, i):
    word, pos, _ = sentence_with_tags[i]

    features = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.has_hyphen': '-' in word,
        'postag': pos,
    }

    if i > 0:
        word1, postag1, _ = sentence_with_tags[i - 1]
        features.update({
            '-1:word.lower': word1.lower(),
            '-1:postag': postag1,
            '-1:word.istitle': word1.istitle(),
            '-1:word.isupper': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sentence_with_tags) - 1:
        word1, postag1, _ = sentence_with_tags[i + 1]
        features.update({
            '+1:word.lower': word1.lower(),
            '+1:postag': postag1,
            '+1:word.istitle': word1.istitle(),
            '+1:word.isupper': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


# Pre-processing the data
def prepare_dataset(tagged_sents):
        """Prepare the dataset by extracting features and labels."""
        X = []
        y = []
        for sentence in tagged_sents:
            features = []
            labels = []
            for index in range(len(sentence)):
                features.append(extract_features(sentence, index))
                labels.append(sentence[index][2])
            X.append(features)
            y.append(labels)
        return X, y

X,y=prepare_dataset(sentences)


# Training the CRF Model
crf = CRF(algorithm = 'lbfgs',
         c1 = 0.1,
         c2 = 0.1,
         max_iterations = 100,
         all_possible_transitions = True)
crf.fit(X, y)








