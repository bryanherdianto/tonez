# Tonez

A TensorFlow.js sentiment analysis demo that reads the tone of a restaurant review, right in your browser.

Link: https://bryanherdianto.github.io/sentiment-website/

## Features

- **Live sentiment scoring** — type a review and get an instant positive/negative score
- **Quick-try examples** — one-click sample reviews to try the demo immediately
- **Session history** — every review you test is logged below the demo for easy comparison
- **About the Model** — a look at the dataset, preprocessing, and architecture behind the demo

## Tech

- [TensorFlow.js](https://www.tensorflow.org/js) for in-browser inference
- Model trained in Python with TensorFlow/Keras on 1,000 labeled Yelp reviews (see
  `notebook.ipynb`), then converted to a TensorFlow.js layers model (`tfjs_model/`)
- Plain HTML/CSS/JS front end — no build step or framework
