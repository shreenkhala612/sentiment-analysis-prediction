from flask import Flask,render_template,request
import joblib
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

app=Flask(__name__)
          #template_folder='template')   
@app.route('/')
def Home():
    return render_template("HomePage.html")
	
@app.route('/predict',methods=['POST'])
def predict():
    def openFile(path):
        with open(path) as file:
            data = file.read()
        return data

    imdb_data = openFile('imdb_labelled.txt')
    amzn_data = openFile('amazon_cells_labelled.txt')
    yelp_data = openFile('yelp_labelled.txt')
    datasets = [imdb_data, amzn_data, yelp_data]
    combined_dataset=[]
    for dataset in datasets:
        combined_dataset.extend(dataset.split('\n'))
    dataset = [sample.split('\t') for sample in combined_dataset]
    df = pd.DataFrame(data=dataset, columns=['Reviews', 'Labels'])
    df.head()
    df.count()
    # Remove any blank reviews
    df = df[df["Labels"].notnull()]
    df = df.sample(frac=1)
    
    df['Word Count'] = [len(review.split()) for review in df['Reviews']]
    
    
    df['Uppercase Char Count'] = [sum(char.isupper() for char in review) \
                                  for review in df['Reviews']]                           
    
    df['Special Char Count'] = [sum(char in string.punctuation for char in review) \
                                for review in df['Reviews']]     
    df['Word Count'].describe()
    negative_samples=df[df.Labels=='0']
    positive_samples=df[df.Labels=='1']
    positive_samples['Word Count'].describe()
    negative_samples['Word Count'].describe()
    
    def getMostCommonWords(reviews, n_most_common, stopwords=None):
       
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
    
    vectorizer = TfidfVectorizer(min_df=15)
    bow = vectorizer.fit_transform(df['Reviews'])
    labels = df['Labels']
    print(len(vectorizer.get_feature_names()))
    
    X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.33)
    
    classifier = rfc()
    classifier.fit(X_train,y_train)
    print("Initial classifier score is")
    print(classifier.score(X_test,y_test))
    
    if request.method == 'POST':
            review=request.form['main_input']
            data=[review]
            classifier.fit(bow, labels)
            vect=vectorizer.transform(data).toarray()
            vecto = vectorizer.transform(data)
            my_pre= classifier.predict_proba(vecto)
            my_prediction = classifier.predict(vect)
            op1=my_pre[0][1]
            op2=my_pre[0][0]
            if(op1>op2):
                my_prediction1 = op1-op2
                pre=1
            else:
                my_prediction1 = op2-op1
                pre=0

    return render_template('HomePage.html', prediction=my_prediction1 ,
                           output=op1, out=op2, pred=pre)

if __name__ == "__main__":
    app.run(debug=True)