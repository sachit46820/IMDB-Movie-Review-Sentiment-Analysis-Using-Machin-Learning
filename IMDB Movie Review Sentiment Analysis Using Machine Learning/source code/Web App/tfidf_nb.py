# from flask import Flask,request,render_template
# import pickle
#
# # app = Flask(__name__)
# tfidf_vec = pickle.load(open('tfidf_file', 'rb'))
# model = pickle.load(open('nb_tfidf (1)', 'rb'))
#

# @app.route("/")
# def index():
#     return render_template("index.html")
#


# @app.route("/movieReview1",methods=['post'])
# def movieReview1():
#     review = request.form.get("review")
#     result = model.predict(tfidf_vec.transform([review]))
#     if result == 0:
#         return render_template("message.html", msg='positive', color='bg-success')
#     else:
#         return render_template("message.html", msg='negative', color='bg-warning')

# app.run(debug=True)