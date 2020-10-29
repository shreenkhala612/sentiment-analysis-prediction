from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import string
import numpy as np
import itertools
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import matplotlib.pyplot as plt

def openFile(path):
    #param path: path/to/file.ext (str)
    #Returns contents of file (str)
    with open(path) as file:
        data = file.read()
    return data

imdb_data = openFile('imdb_labelled.txt')
amzn_data = openFile('amazon_cells_labelled.txt')
yelp_data = openFile('yelp_labelled.txt')


datasets = [imdb_data, amzn_data, yelp_data]
combined_dataset=[]
# separate samples from each other
for dataset in datasets:
    combined_dataset.extend(dataset.split('\n'))
print("Combined dataset \n",combined_dataset[1:2])
# separate each label from each sample
dataset = [sample.split('\t') for sample in combined_dataset]
print("New dataset \n",dataset[1:2])

df = pd.DataFrame(data=dataset, columns=['Reviews', 'Labels'])
df.head()
df.count()
# Remove any blank reviews
df = df[df["Labels"].notnull()]
        


# shuffle the dataset for later.
# Note this isn't necessary (the dataset is shuffled again before used), 
# but is good practice.
df = df.sample(frac=1)

df['Word Count'] = [len(review.split()) for review in df['Reviews']]


df['Uppercase Char Count'] = [sum(char.isupper() for char in review) \
                              for review in df['Reviews']]                           

df['Special Char Count'] = [sum(char in string.punctuation for char in review) \
                            for review in df['Reviews']]     
df['Word Count'].describe()
negative_samples=df[df.Labels=='0']
print("Negative samples \n")
print(negative_samples.head())
positive_samples=df[df.Labels=='1']
print("Positive samples \n")
print(positive_samples.head())

positive_samples['Word Count'].describe()
negative_samples['Word Count'].describe()

def getMostCommonWords(reviews, n_most_common, stopwords=None):
    # param reviews: column from pandas.DataFrame (e.g. df['Reviews']) 
        #(pandas.Series)
    # param n_most_common: the top n most common words in reviews (int)
    # param stopwords: list of stopwords (str) to remove from reviews (list)
    # Returns list of n_most_common words organized in tuples as 
        #('term', frequency) (list)

    # flatten review column into a list of words, and set each to lowercase
    flattened_reviews = [word for review in reviews for word in \
                         review.lower().split()]


    # remove punctuation from reviews
    flattened_reviews = [''.join(char for char in review if \
                                 char not in string.punctuation) for \
                         review in flattened_reviews]


    # remove stopwords, if applicable
    if stopwords:
        flattened_reviews = [word for word in flattened_reviews if \
                             word not in stopwords]


    # remove any empty strings that were created by this process
    flattened_reviews = [review for review in flattened_reviews if review]

    return Counter(flattened_reviews).most_common(n_most_common)
#Without removing stopwords
print(getMostCommonWords(positive_samples['Reviews'], 10))
#After removing stopwords
print(getMostCommonWords(positive_samples['Reviews'], 10, stopwords.words('english')))
print(getMostCommonWords(negative_samples['Reviews'], 10, stopwords.words('english')))


######%%%%%
#Begin vectorising

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
bow = vectorizer.fit_transform(df['Reviews'])
labels = df['Labels']
print("____________________________________________________________________")
print("Initial number of features in our dataset:")
print(len(vectorizer.get_feature_names()))

vectorizer = TfidfVectorizer(min_df=15)
bow = vectorizer.fit_transform(df['Reviews'])
labels = df['Labels']
print("____________________________________________________________________")
print("Optimising the Bag Of Words to frequently used words:")
print("Resulting BOW")

print(vectorizer.get_feature_names())
print("Optimised Length")
print(len(vectorizer.get_feature_names()))
feature_names = vectorizer.get_feature_names()

###########



##wordcloud
from wordcloud import WordCloud, STOPWORDS
text1 = " ".join(review for review in positive_samples.Reviews)
text2 = " ".join(review for review in negative_samples.Reviews)
stopwords = set(STOPWORDS)
# Generate a word cloud image
wordcloud1 = WordCloud(stopwords=stopwords,width=800, height=800, max_words=50,
                     background_color='white',
                      min_font_size=10).generate(text1)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud1,interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

wordcloud2 = WordCloud(stopwords=stopwords,width=800, max_words=50,height=800,
                     background_color='white',
                      min_font_size=10).generate(text2)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud2,interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()



#training the model
X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.33)
print("____________________________________________________________________")
print("Training the model using Random Forest Classification Method")


#Random forest
classifier = rfc()
classifier.fit(X_train,y_train)
print("RFC classifier score is")
print(classifier.score(X_test,y_test))
y_pred=classifier.predict(X_test)




#Confusion matrix for initial RFC
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['neg','pos'],
                      title='Initial RFC Classifier\n')


plt.show()
print("Accuracy Score=")
print(accuracy_score(y_test, y_pred));


###############
print("____________________________________________________________________")
print("Hyperparameter Optimizatio")
classifier = rfc()

hyperparameters = {
    'n_estimators':stats.randint(10,300),
    'criterion':['gini','entropy'],
    'min_samples_split':stats.randint(2,9),
    'bootstrap':[True,False]
}

random_search = RandomizedSearchCV(classifier, hyperparameters, n_iter=65, n_jobs=4)

random_search.fit(bow, labels)

optimized_classifier = random_search.best_estimator_
optimized_classifier.fit(X_train,y_train)

print("optimized classifier score is  :")
print(optimized_classifier.score(X_test,y_test))
y_pred=optimized_classifier.predict(X_test)


#Confusion matrix for Hyperparameter Optimization
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['neg','pos'],
                      title='Optimized  Classifier\n')

plt.show()
print("Accuracy Score=")
print(accuracy_score(y_test, y_pred));
##CHECKING

optimized_classifier.fit(bow,labels)

our_negative_sentence = vectorizer.transform(['I hated this product. It is \
not well designed at all, and it broke into pieces as soon as I got it. \
Would not recommend anything from this company.'])

our_positive_sentence = vectorizer.transform(['The movie was superb - I was \
on the edge of my seat the entire time. The acting was excellent, and the \
scenery - my goodness. Watch this movie now!'])
print("Probablities of the sentences being positive or negative")
print("1st Sentence:", optimized_classifier.predict_proba(our_negative_sentence))
print("2nd Sentence:", optimized_classifier.predict_proba(our_positive_sentence))


