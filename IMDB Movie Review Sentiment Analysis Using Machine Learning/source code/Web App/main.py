from flask import Flask,request,render_template
import pickle


tfidf_vec = pickle.load(open('tfidf_file', 'rb'))
model_nb_tfidf = pickle.load(open('nb_tfidf (1)', 'rb'))

nb_bow_vec = pickle.load(open('countvec_file', 'rb'))
model_nb_bow = pickle.load(open('nb_bow', 'rb'))

sgd_bow_vec = pickle.load(open('countvec_file', 'rb'))
model_sgd_bow = pickle.load(open('sgd_bow', 'rb'))

sgd_tfidf_vec = pickle.load(open('tfidf_file', 'rb'))
model_sgd_tfidf = pickle.load(open('sgd_tfidf', 'rb'))


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/bagOfWords1")
def bagOfWords1():
    return render_template("bagOfWords1.html")

@app.route("/bagOfWords2")
def bagOfWords2():
    return render_template("bagOfWords2.html")

@app.route("/TFIDF1")
def TFIDF1():
    return render_template("TFIDF1.html")

@app.route("/TFIDF2")
def TFIDF2():
    return render_template("TFIDF2.html")



@app.route("/bagOfWordsReview1",methods=['post'])
def bagOfWordsReview1():
    review = request.form.get("review")
    result = model_nb_bow.predict(tfidf_vec.transform([review]))
    if result == 0:
        return render_template("message.html", msg='Bag of Words 1 result : Naive Bayes Result is Positive', color='bg-success')
    else:
        return render_template("message.html", msg='Bag of Words 1 result :Naive Bayes Result is Negative', color='bg-danger')


@app.route("/bagOfWordsReview2",methods=['post'])
def bagOfWordsReview2():
    review = request.form.get("review")
    result = model_sgd_bow.predict(tfidf_vec.transform([review]))
    if result == 0:
        return render_template("message.html", msg='Bag of Words 2 result : Stochastic Gradient is Positive', color='bg-success')
    else:
        return render_template("message.html", msg='Bag of Words 2 result : Stochastic Gradient is Negative', color='bg-danger')


@app.route("/TFiDfReview1",methods=['post'])
def TFiDfReview1():
    review = request.form.get("review")
    result = model_nb_tfidf.predict(tfidf_vec.transform([review]))
    if result == 0:
        return render_template("message.html", msg='TFiDf 1 result : Naive Bayes Result is  Positive', color='bg-success')
    else:
        return render_template("message.html", msg='TFiDf 1 result : Naive Bayes Result is Negative', color='bg-danger')




@app.route("/TFiDfReview2",methods=['post'])
def TFiDfReview2():
    review = request.form.get("review")
    result = model_sgd_tfidf.predict(tfidf_vec.transform([review]))
    if result == 0:
        return render_template("message.html", msg='TFiDf 2 result :  Stochastic Gradient is  Positive', color='bg-success')
    else:
        return render_template("message.html", msg='TFiDf 2 result : Stochastic Gradient is Negative', color='bg-danger')






app.run(debug=True)