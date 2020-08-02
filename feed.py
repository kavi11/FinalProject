from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pymongo import MongoClient
import pandas as pd
from textblob import TextBlob
from flask import Flask, render_template
from flask_cors import CORS
import requests
import re
import csv
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pytesseract
from imageai.Prediction import ImagePrediction
import os

# DataBase Connection
client = MongoClient("mongodb+srv://kavi:kavi123@cluster0-wovzx.mongodb.net/BBC?retryWrites=true&w=majority")
db = client.CBC

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/image')
def image():
    execution_path = os.getcwd()
    print(execution_path)
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath( execution_path + "/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    prediction.loadModel()

    predictions, percentage_probabilities = prediction.predictImage("car.jpeg", result_count=3)
    predictions1, percentage_probabilities1 = prediction.predictImage("news11.jpg", result_count=3)
    predictions2, percentage_probabilities2 = prediction.predictImage("airplane.jpg", result_count=3)
    image_items = []
    for index in range(len(predictions)):
            image_item = {}
            image_item['Index'] = predictions[index]
            image_item['Prediction'] = percentage_probabilities[index]
            print(predictions[index] , " : " , percentage_probabilities[index])
            # image_items.append(image_item)
    for index in range(len(predictions1)):
            image_item = {}
            image_item['Index'] = predictions1[index]
            image_item['Prediction'] = percentage_probabilities1[index]
            print(predictions1[index], " : ", percentage_probabilities1[index])
    for index in range(len(predictions1)):
            image_item = {}
            image_item['Index'] = predictions2[index]
            image_item['Prediction'] = percentage_probabilities2[index]
            print(predictions2[index], " : ", percentage_probabilities2[index])

            return  render_template('imagerec.html',p=predictions[index],p1=percentage_probabilities[index],
                                    p2=predictions1[index],p3=percentage_probabilities1[index],
                                    p4=predictions2[index],p5=percentage_probabilities2[index])

@app.route('/CBC')
def CBC():
    # link for rss feeds
    # https://www.cnn.com/services/rss/
    CBC_url = "https://www.cbc.ca/cmlink/rss-technology"
    CBC_get = requests.get(CBC_url)
    CBC_soup = BeautifulSoup(CBC_get.content, features="xml")
    CBC = CBC_soup.findAll('item')
    # print(BBC)

    CBC_items = []
    for item in CBC:
        CBC_item = {}
        CBC_item['Title'] = item.title.text
        CBC_item['Description'] = item.description.text
        CBC_item['Link'] = item.link.text
        CBC_item['Date'] = item.pubDate.text

        blob = TextBlob(item.title.text)

        print(blob.sentiment.polarity)
        value = blob.sentiment.polarity

        if blob.sentiment.polarity < 0:
            bbc_sentiment = 'Negative'

            print("Negative")
        elif blob.sentiment.polarity > 0:
            print("Positive")
            bbc_sentiment = 'Positive'
        else:
            print("Neutral")
            bbc_sentiment = 'Neutral'

        CBC_item['Sentiment'] = bbc_sentiment

        CBC_item['Polarity'] = value
        CBC_items.append(CBC_item)

    Dataframe_CBC = pd.DataFrame(CBC_items, columns=['Title', 'Link', 'Date','Sentiment','Polarity'])
    # print(Dataframe_CBC)
    Dataframe_CBC.to_csv('data.csv', index=False, encoding='utf-8')
    db.feed_CBC.insert_many(Dataframe_CBC.to_dict('records'))
    return render_template('CBC.html', tables=[Dataframe_CBC.to_html(classes='data')], titles=Dataframe_CBC.columns.values)



@app.route('/imagerec')
def imagerec():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    img = Image.open('7.png')
    img1 = Image.open('NEW111.jpg')
    img2 = Image.open('14.png')
    img3 = Image.open('11.png')
    img4 = Image.open('222.png')

    textapple=pytesseract.image_to_string(img,config='preserve_interword_spaces=1')
    textapple1 = pytesseract.image_to_string(img1, config='preserve_interword_spaces=1')
    textapple2 = pytesseract.image_to_string(img2, config='preserve_interword_spaces=1')
    textapple3 = pytesseract.image_to_string(img3, config='preserve_interword_spaces=1')
    textapple4 = pytesseract.image_to_string(img4, config='preserve_interword_spaces=1')
    print(textapple)
    print(textapple1)
    print(textapple2)
    print(textapple3)
    print(textapple4)
    return render_template('image.html', a=textapple,b=textapple1,c=textapple2,d=textapple3,e=textapple4)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
