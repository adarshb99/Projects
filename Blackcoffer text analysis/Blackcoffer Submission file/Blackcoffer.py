#!/usr/bin/env python
# coding: utf-8

# In[2]:


#we first import all the required libraries
import pandas as pd
from bs4 import BeautifulSoup
import requests


# In[3]:


#we read the dataset provided
df=pd.read_excel('Input.xlsx')
df.head()


# In[ ]:


#we now extract the article title and article text as mentioned
#I have used beautiful soup for data crawling.


# In[7]:


#we will extract the article title and the text

def extract_article_text(url):
    try:
        response=requests.get(url)
        soup=BeautifulSoup(response.text,'html.parser')
        
        article_text=' '.join([p.get_text() for p in soup.find_all('p')])
        
        return article_text
    
    except Exception as e:
        print(f'error extracting text from {url}:{e}')
        return None
        
for index,row in df.iterrows():
    url_id=row['URL_ID']
    url=row['URL']
    
    article_text=extract_article_text(url)
    
    if article_text:
        output_file=f'{url_id}.txt'
        with open(output_file,'w',encoding='utf-8') as file:
            file.write(article_text)
            
        print(f"Text extracted from {url} and saved to {output_file}")
    else:
        print(f"Skipping {url_id} due to extraction error")

print("Extraction process completed.")
    
        
        

In the above code I have extracted data from the given links and saved it in a 'txt' file format with
name as the 'URL_ID' and its content respectively. Thus the extraction process is completed.
# # Data Analysis

# In[4]:


#Here we start the text analysis
#We first import all the necessary libraries and create
#a dataframe to store values.


# In[7]:


import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize


input_file = 'Input.xlsx'
df_input = pd.read_excel(input_file)

#creating a dataframe according to the output file
columns = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
           'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
           'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
           'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
df_output = pd.DataFrame(columns=columns)

#To avoid confusion I have extracted text from the article again
#Sentimental analysis is done using textblob library
#We convert the text into a list of tokens using the nltk tokenize module and use these tokens
#to calculate the 4 variables which are positive score, negative score, polarity score, subjectivity score
# In[8]:


# Function to extract article text from a given URL
def extract_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Assuming the article text is contained within 'p' tags. You may need to inspect the HTML structure of the websites.
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])

        return article_text

    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return None

# Function to perform text analysis and populate the DataFrame
def perform_text_analysis(text):
    # Sentiment analysis
    blob = TextBlob(text)
    positive_score = blob.sentiment.polarity
    negative_score = 1 - positive_score  # Assuming a range of [0, 1] for positive sentiment
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity

    # Other text metrics (customize as needed)
    words = word_tokenize(text)
    avg_sentence_length = len(words) / len(sent_tokenize(text))
    percentage_of_complex_words = len([word for word in words if len(word) > 6]) / len(words)
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)
    avg_words_per_sentence = len(words) / len(sent_tokenize(text))
    complex_word_count = len([word for word in words if len(word) > 6])
    word_count = len(words)
    syllable_per_word = sum([syllables(word) for word in words]) / len(words)
    personal_pronouns = count_personal_pronouns(text)
    avg_word_length = sum(len(word) for word in words) / len(words)

    # Return computed values
    return [positive_score, negative_score, polarity_score, subjectivity_score,
            avg_sentence_length, percentage_of_complex_words, fog_index,
            avg_words_per_sentence, complex_word_count, word_count,
            syllable_per_word, personal_pronouns, avg_word_length]

# Function to count personal pronouns
def count_personal_pronouns(text):
    personal_pronouns_list = ['I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourselves', 'they', 'them', 'their', 'theirs', 'themselves']
    words = word_tokenize(text)
    personal_pronouns_count = sum(1 for word in words if word.lower() in personal_pronouns_list)
    return personal_pronouns_count

# Function to count syllables in a word
def syllables(word):
    return sum([1 for char in word.lower() if char in 'aeiou'])

# Iterate through the URLs in the DataFrame and perform text analysis
for index, row in df_input.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Extract article text
    article_text = extract_article_text(url)

    if article_text:
        # Perform text analysis
        analysis_results = perform_text_analysis(article_text)

        # Create a dictionary with the results
        analysis_dict = {'URL_ID': url_id, 'URL': url}
        analysis_dict.update(dict(zip(columns[2:], analysis_results)))

        # Append the results to the output DataFrame
        df_output = df_output.append(analysis_dict, ignore_index=True)

        print(f"Text analysis completed for {url}")

# Save the output DataFrame with text analysis metrics to a new Excel file
output_file = 'Output.xlsx'
df_output.to_excel(output_file, index=False)

print(f"Text analysis results saved to {output_file}")

Project instructions

Overview:
This project involves extracting textual data from a list of URLs, performing text analysis, 
and computing various variables based on the specified output structure. The computed results 
are saved in an Excel file in the same order as the output structure file.

Approach:
Data Collection:

URLs are provided in an Excel file (Input.xlsx).
Python script uses web scraping (BeautifulSoup) to extract article text from each URL.

Text Analysis:

Text analysis is performed on the extracted article text.
Sentiment analysis is conducted using the TextBlob library.
Additional metrics such as average sentence length, percentage of complex words, and Fog Index
are computed.

Dependencies:

Ensure you have the necessary Python libraries installed such as beautiful soup, nltk, pandas.

How to Run:

Save the URLs in an Excel file named Input.xlsx with columns 'URL_ID' and 'URL'.
Run the Python script (script.py) to perform the data extraction and analysis:
bash

python script.py
The results will be saved in an Excel file named Output.xlsx.
Output Structure:

The output file (Output.xlsx) contains the computed variables in the exact order specified in the 
output structure file.
# In[ ]:




