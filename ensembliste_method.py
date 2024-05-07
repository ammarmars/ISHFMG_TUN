#!/usr/bin/env python
#import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix as confmat
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import re
from sklearn.ensemble import GradientBoostingClassifier
import string
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE

#load the training data
data=pd.read_csv('data/Mawqif_AllTargets_Train.csv')

#load the test blind data
data_blind_test=pd.read_csv('data/Mawqif_AllTargets_Blind Test.csv')

data_stance=data[['text','stance']]
#delete the missed values 
data_stance=data_stance.fillna('None')
#mapping the labes 
mapping = {'None':0,'Favor': 1, 'Against': 2}
data_stance['stance'] = data_stance['stance'].apply(lambda x: mapping[x])

'''
The first step is to subject the data to preprocessing.
This involves removing both arabic and english punctuation
Normalizing different letter variants with one common letter
'''
# first we define a list of arabic and english punctiations that we want to get rid of in our text

punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation

# Arabic stop words with nltk
stop_words = stopwords.words()

arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def preprocess(text):
    
    '''
    text is an arabic string input
    
    the preprocessed text is returned
    '''
    
    #remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)
    
    # remove Tashkeel
    text = re.sub(arabic_diacritics, '', text)
    #remove no arabic words
    text= re.sub('[a-zA-Z0-9_]|#|http\S+', '', text)
    
    #remove longation
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text
  


#preprocessing the training data
data_stance['text'] = data_stance['text'].apply(preprocess)

#preprocessing the blind test data
data_blind_test['text']=data_blind_test['text'].apply(preprocess)

# assign the text to a variable (X)
X=data_stance['text']
# assign the stance to a variable (y)
y=data_stance['stance']

# assign the text of blind test data to a variable (X_blind_test)
X_blind_test=data_blind_test['text']

#define the list of stance names
target_names=['NONE','FAVOR','AGAINST']


def basic_tokenize(tweet):
    return tweet.split(' ')


def skipgram_tokenize(tweet, n=None, k=None, include_all=True):
    from nltk.util import skipgrams
    tokens = [w for w in basic_tokenize(tweet)]
    if include_all:
        result = []
        for i in range(k+1):
            skg = [w for w in skipgrams(tokens, n, i)]
            result = result+skg
    else:
        result = [w for w in skipgrams(tokens, n, k)]
    result=set(result)
    return result


def make_skip_tokenize(n, k, include_all=True):
    return lambda tweet: skipgram_tokenize(tweet, n=n, k=k, include_all=include_all)


# features extraction using FeatureUnion
max_df = 0.5
union = FeatureUnion([("w_v", TfidfVectorizer(sublinear_tf=True, max_df=max_df,analyzer = 'word', ngram_range=(1,5)
                                 )),
                       ("c_wb", TfidfVectorizer(sublinear_tf=True,max_df=max_df,analyzer = 'char_wb', ngram_range=(1,5)
                                 )),
                       ("c_wb5", TfidfVectorizer(sublinear_tf=True, max_df=max_df,analyzer = 'char', ngram_range=(1,4)
                                 )),

      ("sk",TfidfVectorizer(sublinear_tf=True, max_df=max_df,tokenizer=make_skip_tokenize(n=2, k=1)))

                       ],
transformer_weights={
            'w_v': 0.5,
            'c_wb': 0.5,
           ' c_wb5':0.5,
            'sk': 0.3,
        }
,
)

#Transform the text to numeric vector using union feature
X=union.fit_transform(X)
X_blind_test=union.transform(X_blind_test)
  

print(X.shape)
print(X_blind_test.shape)



# Instantiate the SMOTE object
smote = SMOTE()

# Perform oversampling
X_oversampled, y_oversampled = smote.fit_resample(X, y)

#Train Test split
X_train, X_test, y_train, y_test = train_test_split(X_oversampled,
                                                  y_oversampled,
                                                  test_size=0.15,
                                                  random_state=17,stratify=y_oversampled)

# training our model using ensemblist method
estimators = []
 
sgd = SGDClassifier(alpha=0.00001, max_iter=50,penalty="l2") 
estimators.append(('sgd', sgd))
svc = LinearSVC(penalty='l2', dual=False,tol=1e-3)
estimators.append(('svc',svc))
mnb= MultinomialNB(alpha=.01)
estimators.append(('mnb',mnb))
rec=RidgeClassifier()
estimators.append(('rec',rec))
 
RF=RandomForestClassifier()
estimators.append(('RF',RF))

ensemble = VotingClassifier(estimators)
ensemble.fit(X_oversampled, y_oversampled)
#predict stance
pred = ensemble.predict(X_test)

#compute accuracy
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


#display the classification report
print("classification report:")
print(metrics.classification_report(y_test, pred,target_names=target_names))
cm = confmat(y_test, pred)

#predict stance on the blind test data
pred_blind=ensemble.predict(X_blind_test)

#create the prediction labels for the blind test file with the adequate format
ID_blind=data_blind_test['ID'].tolist()
target_blind=data_blind_test['target'].tolist()
teweet_blind=data_blind_test['text'].tolist()

cat_blind=[]
for pred in pred_blind:
    cat_blind.append(target_names[pred])


#create the dataframe 
df=pd.DataFrame(list(zip(ID_blind,target_blind,teweet_blind,cat_blind)),columns =['ID','Target','Tweet','Stance'])

#convert the dataframe to the csv file
df.to_csv('result/guessFile.csv',sep='\t',index=False)


