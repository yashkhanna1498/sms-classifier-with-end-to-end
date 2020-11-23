from flask import Flask, render_template, request
import pickle


filename = 'sms_spam.pkl'
classifier = pickle.load(open(filename, 'rb'))
pro = pickle.load(open('text_processor.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        msg=[message]
        v= pro.transform(msg).toarray()
        prediction=classifier.predict(v)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
	app.run(debug=False)