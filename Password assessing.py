#import libraries requird
#should use Jupyter notebook to run this code
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

#Get a custom tokenizer to break down the word rather then taking the full text
def getTokens(inputString): 
	tokens = []
	for i in inputString:
		tokens.append(i)
	return tokens
  
#path for password file for training
filepath = 'your_file_path_containing_passwords_and_labels' 
data = pd.read_csv(filepath,',',error_bad_lines=False)

data = pd.DataFrame(data)
passwords = np.array(data)

#check for strenth by shuffling and check it against the actual password
random.shuffle(passwords) 
y = [d[1] for d in passwords]
#actual passwords 
allpasswords= [d[0] for d in passwords] 

vectorizer = TfidfVectorizer(tokenizer=getTokens) #vectorizing
X = vectorizer.fit_transform(allpasswords)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#our logistic regression classifier
lgs = LogisticRegression(penalty='l2',multi_class='ovr')  
lgs.fit(X_train, y_train) #training
print(lgs.score(X_test, y_test))  #testing

#more testing against the database 
X_predict = ['faizanahsdfgmad','faizanahmsdfgad123','ajd1348#sdfg28t**','kuiqw45454asdi','uiquiui455424','mynamei454s123faizan#','Ahmsdfgsded','12dsfgsd3456','abcdefasdfaa455']
X_predict = vectorizer.transform(X_predict)
y_Predict = lgs.predict(X_predict)
print y_Predict
