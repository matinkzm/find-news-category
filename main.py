import requests
import csv
from bs4 import BeautifulSoup
import json
import math as m
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import nltk  # used for natural language processing
from sklearn.model_selection import train_test_split  # used to create train and test model for ML
from nltk.corpus import stopwords  # used to ignore stopwords like the,a,an,in and so on
from nltk.tokenize import word_tokenize  # used to tokenize(syllables) a string
from wordcloud import WordCloud  # create a box that contains related words that are in a same category

nltk.download('stopwords')  # download stopwords e.g.: a, an , and, are, as, at, be, but
nltk.download('punkt')  # divides text to into sentence or word

stop_words = set(stopwords.words("english"))

# create a category list because request method does not get all news
category_list = ["lifestyle", "lifestyle/tagged/health/", "news/politics/", "news/science/", "lifestyle/tagged/food/",
                 "entertainment/movies/", "lifestyle/tagged/travel/"]
# base url
url = "https://www.yahoo.com/"
# get the news and save it into a csv file
with open('EN_news.csv', 'a', newline='') as csvfile:
    for item in category_list:
        # check for request response
        req = requests.get(url + item)
        # print(req)

        # parse data into html
        html_result = BeautifulSoup(req.content, "html.parser")
        # print(html_result.prettify())

        # remove redundant parts
        for h3s in html_result.find_all("h3"):
            if len(h3s.text) > 20:
                csvfile.writelines(h3s.text + "\n")





# get words after filtering stop words
# remove stopwords from text
def get_filtered_words(headline):
    return [word for word in word_tokenize(headline) if not word in stop_words]


# Build a vocab list from train dataset
def build_vocab_list(train_dataset):
    vocab_list = []
    for record in train_dataset:
        filtered_words = get_filtered_words(record['headline'])
        vocab_list.extend(filtered_words)
    return vocab_list


# create count frequency of words on training data
def build_doc_freq_dict(train_dataset, categories):
    word_frequency_dict = {category: {vocab: 0 for vocab in vocab_list} for category in categories}
    for record in train_dataset:
        for word in get_filtered_words(record['headline']):
            word_frequency_dict[record['category']][word] += 1
    return word_frequency_dict


# separate dataset by class
def get_data_group_by_classes(dataset):
    seperated_category_dict = {category: [] for category in categories}
    for record in dataset:
        seperated_category_dict[record['category']].append(record)
    return seperated_category_dict


# checks if smoothing is needed
def is_smoothing_needed(filtered_words, word_dict, category):
    for word in filtered_words:
        if word not in word_dict[category].keys():
            return True
    return False


# get probability of the text assuming conditional independence(naive bayes)
def get_naive_probability(category, text, word_dict, grouped_classes_dict, dataset):
    if category not in word_dict.keys():
        return
    total_word_cond_prob = 1
    smoothing_factor = 0.001
    filtered_words = get_filtered_words(text)
    smoothing_indicator = is_smoothing_needed(filtered_words, word_dict, category)
    for word in filtered_words:
        if smoothing_indicator:
            word_doc_freq = word_dict[category][word] if word in word_dict[category].keys() else 0
            total_word_cond_prob *= ((word_doc_freq + 1) / ((len(grouped_classes_dict[category]) + (
                    smoothing_factor * len(grouped_classes_dict[category])))))  # Smoothing technique
        else:
            total_word_cond_prob *= (word_dict[category][word] / len(grouped_classes_dict[category]))
    class_probability = (len(grouped_classes_dict[category]) / len(dataset))
    return total_word_cond_prob * class_probability


# get the predicted class given text belong to
def get_predicted_class(categories, str_to_predict, word_frequency_dict, group_classes_dict, train_dataset):
    greatest_prob = 0
    predicted_category = ''
    for category in categories:
        total_word_cond_prob = 1
        prob_of_category_contains_headline = get_naive_probability(category, str_to_predict, word_frequency_dict,
                                                                   group_classes_dict, train_dataset)
        if prob_of_category_contains_headline > greatest_prob:
            greatest_prob = prob_of_category_contains_headline
            predicted_category = category

    return predicted_category


# calculate the accuracy of the classifier over given dataset:
def get_accuracy(dataset, categories, word_frequency_dict, group_classes_dict):
    correct_predicted = 0
    for record in dataset:
        predicted_category = get_predicted_class(categories, record['headline'], word_frequency_dict,
                                                 group_classes_dict, dataset)
        if predicted_category == record['category']:
            correct_predicted += 1
    return correct_predicted / len(dataset)


def get_top_words(word_frequency_dict, category):
    sorted_list = sorted(word_frequency_dict[category].items(), key=lambda item: item[1], reverse=True)
    top_words = {word: count for word, count in sorted_list if len(word) > 2}
    return top_words


def print_top_words(word_frequency_dict, categories):
    plot.figure(figsize=(18, 30))
    for index, category in enumerate(categories, start=1):
        words = get_top_words(word_frequency_dict, category)
        wordcloud = WordCloud(width=480, height=480, background_color='white',
                              min_font_size=14)
        wordcloud.generate_from_frequencies(words)
        # plot the WordCloud image
        plot.subplot(8, 5, index)
        plot.imshow(wordcloud, interpolation="bilinear")
        plot.title(category)
        plot.axis("off")
    plot.show()


# sample news file
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
# print(df.head().to_string())
# print(df.category.unique())

# Load the original dataset into dictionary and  data preprocessing.
dataset = []
categories = []
stop_words = set(stopwords.words("english"))
with open('News_Category_Dataset_v3.json') as data:
    for line_num, file_text in enumerate(data, start=0):
        # do some cleaning for better readability
        file_text = file_text.replace('"category": "CULTURE & ARTS",', '"category": "ARTS & CULTURE",', 1).replace(
            '"category": "WORLDPOST",', '"category": "THE WORLDPOST",', 1).replace('"category": "PARENTING",',
                                                                                   '"category": "PARENTS",', 1)
        record = json.loads(file_text)
        processed_record = {"category": record['category'], "headline": record['headline'].lower()}
        dataset.append(processed_record)
        if dataset[line_num]['category'] not in categories:
            categories.append(dataset[line_num]['category'])

no_of_records = len(dataset)
# print(f""" Number of records in the dataset: {no_of_records}""")
# print(f""" Number of different categories in the dataset: {len(categories)}""")

# Data After Preprocessing
df2 = pd.DataFrame(dataset)
# df2.head()

# plot the data distribution of all news categories in the dataset
fig, ax = plot.subplots()
fig.set_figheight(10)
fig.set_figwidth(10)
ax.barh(df2.loc[:, 'category'].value_counts().keys(), df2.loc[:, 'category'].value_counts().values, alpha=0.5)
ax.set_ylabel("News Categories")
ax.set_xlabel('No.of records in the category')
ax.set_title('News Dataset Classification')
plot.show()

# Divide the dataset into train and test data
k_fold = 1
accuracy_list = []
train_dataset, test_dataset = train_test_split(dataset, test_size=0.1)
# print(f""" Number of records in the training dataset: {len(train_dataset)}""")
# print(f""" Number of records in the test dataset: {len(test_dataset)}""")
# Group all records with their respective category
group_classes_dict = get_data_group_by_classes(train_dataset)
vocab_list = build_vocab_list(train_dataset)
word_frequency_dict = build_doc_freq_dict(train_dataset, categories)
# uncomment to see the accuracy
"""test_accuracy = get_accuracy(test_dataset, categories, word_frequency_dict, group_classes_dict)
print(
    f'''Iteration:{1} Accuracy of the Naive Baye's Classifier over test dataset is {round(test_accuracy * 100, 2)}%''')
accuracy_list.append(test_accuracy)
fig2, ax2 = plot.subplots()
fig2.set_figheight(5)
fig2.set_figwidth(5)
ax2.bar(np.arange(1, 6), accuracy_list, alpha=0.5)
ax2.set_ylabel("Accuracies")
ax2.set_xlabel('Itearions')
ax2.set_title('Accuracies for 5 fold iteration')
plot.show()
print(f'''Average Accuracy of the Naive Baye's Classifier over test dataset is {round(np.mean(np.array(accuracy_list)) * 100, 2)}%''')

print_top_words(word_frequency_dict, categories)"""

# after finding group words for each category and done some test on test and train dataset
# giving my own news to get their categories

final_df = pd.DataFrame(columns=['title', 'category'])
with open("EN_news.csv", "r") as file:
    for row in file:
        temp_category = get_predicted_class(categories, row, word_frequency_dict, group_classes_dict, train_dataset)
        temp_str = str("title: " + row + "category: " + temp_category)
        with open("final_output.csv", "a") as final:
            final.writelines(temp_str + "\n")
        # temp_df = pd.DataFrame([row, temp_category])
        # final_df = pd.concat([final_df, temp_df], ignore_index=True)
        # final_df.loc[len(final_df.index)] = [row, temp_category]

# print(final_df.to_string())



