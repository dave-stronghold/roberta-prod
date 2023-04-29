
from flask import Flask, request
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import os
app = Flask(__name__)


MODEL=f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def analyse(example):
    
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2])
    }


@app.route('/')
def home():
    response_body = {
        'name': 'welcome',
    }

    return response_body



@app.route('/predict_sentiment',methods=['GET', 'POST'])
def d():
    body=request.json
    example= str(body['text'])
    dict=analyse(example)
    response_body = {
        'roberta_neg': dict['roberta_neg'],
        'roberta_neu': dict['roberta_neu'],
        'roberta_pos': dict['roberta_pos']
    }

    return response_body

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))