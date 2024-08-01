# SMS Spam Classification using Naive Bayes

## About the Project

This project provides a web application for classifying SMS messages as spam or ham using a Naive Bayes classifier. The application allows users to enter SMS messages, which are then classified and the result is displayed. The classification model is trained using the SMS Spam Collection Dataset, which is publicly available on [UCI Machine Learning Repository](https://doi.org/10.24432/C5CC84).

![Screenshot](https://github.com/praiseprince/NaiveBayesSMSClassifier/blob/main/Web%20app/static/images/Screenshot.png)

## Files

- **`/Naive Bayes Model`**: This directory contains `model.ipynb` and `sms_dataset.csv`.

- **`/Webapp`**: This directory contains the files for the web app. 

- **`model.ipynb`**: This Jupyter Notebook contains the code for training the Naive Bayes classifier. It processes the SMS dataset, trains the model, and saves the model parameters to `parameters.pkl`. The model achieves an accuracy of 88.43% on the test set.

- **`sms_dataset.csv`**: The SMS Spam Collection Dataset is a collection of SMS messages labeled as "spam" or "ham" (non-spam). It contains a total of 5,572 messages. The proportion of spam messages is 0.1341, and the proportion of ham messages is 0.8659. This dataset is used for evaluating spam detection algorithms and includes diverse examples of both spam and legitimate messages. The dataset is available for download from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5CC84) and is commonly used in machine learning for text classification tasks.

- **`index.html`**: This file provides the user interface for submitting SMS messages and receiving classification results.

- **`main.py`**: This is the main Flask application script that runs the web server. It handles requests from the user, interacts with the trained model, and provides the classification results.

- **`parameters.pkl`**: This file contains the saved parameters of the Naive Bayes model. It is used by the Flask application to make predictions.

- **`requirement.txt`**: This file lists all the Python dependencies required for running the Flask application and training the model.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/praiseprince/Naive-Bayes-SMS-Classifier
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run `main.py` to set up the web app locally.

4. Access the web app by visiting `http://127.0.0.1:5000/` in your browser. 
