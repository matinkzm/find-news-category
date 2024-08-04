# find-news-category
This project is a good practice project for data engineering and data analysis.

The main purpose of this project is to train a machine learning model to find news category base on news headline.

Levels of project:
1.	Import necessary libraries:
2.	Select some random category and get news headlines from yahoo
3.	Remove stop words from news headlines 
4.	Split data into train and test data.
5.	Find top words in each category
6.	For each news headline find most compatible category base on most common number of words in each category.
7.	After doing these works on train dataset and test it with test dataset call functions on data requested from yahoo.


Functions:

build_vocab_list
build a vocabulary list using train dataset 

build_doc_freq_dict
create count frequency of words on training data

get_data_group_by_classes
separate dataset by class

is_smoothing_needed
check if news headline need any smoothing


get_naive_probability
get probability of the text assuming conditional independence(naive bayes)

get_predicted_class
get the predicted class given text belong to

get_accuracy
to test accuracy of guessed category

get_top_words
find top words of each category

print_top_words
plot top words of each category using imshow and wordcloud
