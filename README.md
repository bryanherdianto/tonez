# Tonez

A collection of TensorFlow.js demos that run entirely in your browser. It uses Vanilla HTML, CSS, and JavaScript for the frontend, and TensorFlow.js for in-browser inference. No server and no upload needed because everything runs locally in your browser.

## Pages

- **index.html**: Landing page with an intro and links to both demos.
- **sentiment.html**: Sentiment Analysis demo.
- **gesture.html**: Gesture Shortcut Recognizer demo.

## Demos

### Sentiment Analysis

Reads the tone of a restaurant review, right in your browser. Model is trained in Python with TensorFlow/Keras on 1,000 labeled Yelp reviews (see `sentiment_notebook.ipynb`), then converted to a TensorFlow.js layers model (`tfjs_model_sentiment/`).

**Features**

- **Live sentiment scoring**: type a review and get an instant positive/negative score.
- **Quick-try examples**: one-click sample reviews to try the demo immediately.
- **Session history**: every review you test is logged below the demo for easy comparison.
- **About the Model**: a look at the dataset and preprocessing behind the demo.

### Gesture Shortcut Recognizer

Draw a symbol on a canvas and a small CNN classifies it, right in your browser. Model is trained in Python with TensorFlow/Keras on a **procedurally generated synthetic dataset** — no hand-drawn dataset was collected; each gesture shape is drawn with randomized rotation, scale, position, stroke width, and wobble noise (see `gesture_notebook.ipynb`), then converted to a TensorFlow.js layers model (`tfjs_model_gesture/`).

**Features**

- **Draw-to-classify canvas**: draw a gesture and see it classified as soon as you lift your pointer.
- **Live confidence score**: see how confident the model is in its guess.
- **About the Model**: a look at the (fully synthetic) training data behind the demo.
