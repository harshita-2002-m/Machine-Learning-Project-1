# import libraries
from multiprocessing.spawn import import_main_path
from numpy import append
import pandas as pd
import sklearn
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


# import db
dataset = pd.read_csv(r"E:\\internship\\NLP\\Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

nltk.download('stopwords')

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove("not")
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review= ' '.join(review)
    corpus.append(review)

# create bag of words

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[: , -1].values

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)

# train model using naive bias

gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred= gnb.predict(x_test)

# confusion metrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred))