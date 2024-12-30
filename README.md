# LLM Classification Finetuning

## Table of Contents 
- [Completed by](#completed_by)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Related Works](#related_works)
- [Final Result](#final_result)
- [Conclusion](#conclusion)
- [Sources](#sources)

## COMPLETED BY

<p>BENYAHYA KAWTAR </p>
123456789benyahya@gmail.com

## ABSTRACT

This project uses a Logistic Regression approach with TF-IDF vectorization to discriminate between two text responses in order to determine which response out of the two is better.
The dataset consists of response pairs which were labeled according to the preferred response. 
The text data after pre-processing is vectorized using TFIDF and a logistic regression model is trained to determine the preferred response.
A grid search functionality of GridSearchCV is performed to hypertune the model.
The trained model is used on another dataset to make predictions for the best response and the results are prepared to enter the Kaggle competition “LLM Classification Finetuning”.

## INTRODUCTION

Text responses classification with respect to the user’s taste is a difficult task in the scope of Natural Language Processing (NLP), 
especially after large language models (LLMs) have emerged.
The task of the Kaggle competition “LLM Classification Finetuning” offers an opportunity to compete by fine-tuning LLMs to use human preferences for interactions from conversations with Chatbot Arena. 
For basic text classification problems, there is always a possibility to combine Logistic Regression with TF-IDF vectorization.
This project analyses the success of such methods, which were used during the competition, serving as a threshold level in relation to more sophisticated techniques.

## RELATED WORKS

TF-IDF, or Term Frequency-Inverse Document Frequency, is a popular approach for text vectorization or feature representation of a document by drawing attention to the words
as relative to a collection of the works. Logistic Regression, a method for modeling a binary dependent variable,
has successfully been used in many activities related to text classification such as detection of sentiment and spam .
The use of TF-IDF and Logistic Regression has shown to be effective as a benchmark in most text classification competitions . 
It is worth mentioning that llamas and related models are currently in all the buzz dominating the competition due to their ability to model the language and user traits accurately.
Recent work emphasizes the benefits of fine-tuning strategies for LLMs on complex tasks, on the other hand, 
traditional methods allow us to cover most of the basic tasks with adequate accuracy, but for the complex cases of classification the advanced models could be better.

## FINAL RESULT

<img src="https://github.com/user-attachments/assets/fbd42a4b-4cab-4385-a0d5-8f68bcd3ba9c">
<img src="https://github.com/user-attachments/assets/755380d7-4717-491b-9e5e-405583f52750">
<img src="https://github.com/user-attachments/assets/6691d1f8-7e85-4fe3-b33a-087225d0afd6">
<img src="https://github.com/user-attachments/assets/7e72e46f-38e5-46c2-9893-881c5d89dafd">

## CONCLUSION

Here we present how classical machine learning techniques, namely TF-IDF vectorization with Logistic Regression, 
can be employed to classify pairs of text response in a way that signifies which response a user will select. 
Although this practice can yield a good baseline, the "LLM Classification Finetuning" competition stimulates further sophisticated approaches,
and more particularly the finetuning of large language models toward higher accuracy in the prediction of human preferences. 
More work needs to be done to incorporate and fine-tune LLMs that better reflect the nuances of human language and preferences, 
in keeping with the spirit of the competition and the continued evolution of the NLP landscape.

## SOURCES

- [Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning)
- [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [GeeksforGeeks](https://www.geeksforgeeks.org/text-classification-using-logistic-regression)
- [Kaggle](https://www.kaggle.com/code/kashnitsky/logistic-regression-tf-idf-baseline)
- [Medium](https://medium.com/@ryblovartem/text-classification-baseline-with-tf-idf-and-logistic-regression-2591fe162f3b)
- [Medium](https://medium.com/analytics-vidhya/applying-text-classification-using-logistic-regression-a-comparison-between-bow-and-tf-idf-1f1ed1b83640)
- [GeeksforGeeks](https://www.geeksforgeeks.org/text-classification-using-logistic-regression)
- [Medium](https://drlee.io/text-preprocessing-and-classification-with-logistic-regression-ea4fe3cfcaac)
- [DEV Community](https://dev.to/praveenr2998/countvectorizer-vs-tfidf-logistic-regression-3heb)
- [Stack overflow](https://stackoverflow.com/questions/62766772/using-a-trained-sentiment-analysis-model-tf-idf-and-logistic-regression)
- [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/08/step-by-step-explanation-of-text-classification)
