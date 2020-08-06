# Sentiment-Analysis-on-Reddit-Dataset
Headlines with Python’s Natural Language Toolkit (NLTK)


# The purpose of this project:
In this project We want grouping comments In the Reddit dataset, In three categories: positive, neutral and negative.

# Data Collection
We used Reddit data source for this project.
Reddit is an American social news aggregation, web content rating, and discussion website.
Registered members submit content to the site such as links, text posts, and images, which are then voted up or down by other members.
Each post author is known as a redditor, and each forum dedicated to a specific topic on the website Reddit is known as subreddit.

# Open And Read Dataset
In the first i extracted data and we tried to read And Open the databaseas reddit,
And then we read the data line by line




# Stemming and Lemmatization are Text Normalization (or sometimes called Word Normalization)
techniques in the field of Natural Language Processing that are used to prepare text, words, and documents for further processing. 

# Stemming
Stemming refers to reducing a word to its root form.
While performing natural language processing tasks, you will encounter various scenarios where you find different words with the same root.
For instance, compute, computer, computing, computed, etc.
You may want to reduce the words to their root form for the sake of uniformity.

# Lemmatization
Lemmatization is the process of converting a word to its base form.


# Clean Word for labeling the comments i use SentimentIntensityAnalysis library from nltk. 
this library computes Intensity of each label(neg, neu, pos) for each comment.
this Intensity is based on the words that exist in each comment, After a text is obtained, we start with text normalization. 
Text normalization includes: 
 converting all letters to lower or upper case, 
 converting numbers into words or 
 removing numbers, 
 removing punctuations, 
 accent marks and other diacritics 
 removing white spacesexpanding abbreviations, 
 Remove numbers 
 removing stop words, sparse terms, 
 and particular words, 
 text canonicalization


# Remove punctuation
The following code removes this set of symbols [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:
Remove punctuation and numbers in text,


# What is NLTK?
NLTK stands for Natural Language Toolkit. This toolkit is one of the most powerful NLP libraries which contains packages to make machines understand human language and reply to it with an appropriate response.
Tokenization, Stemming, Lemmatization, Punctuation, Character count, word count are some of these packages.
With Using the library nltk, we downloading List of stopwords and Markings And we download network words.
The project uses the nltk library It is received with the following lines of this library.

# reading the Jason file and calling its preprocessor 
It also clears the input data function And tags In the form of a label :
1 indicates positive , And the 0 tag is a sign of neutrality, And label -1 indicates that the comment is negative.


# With import Matplotlib library 
we show the plot related to the division of comments,
As we can see, most of the comments belong to neutral and show with 0,
The result is a clean data file for further processing.

# NLTK -> Vader
(SentimentIntensityAnalyzer) results with a Machine Learning trained classifier for prediction of Sentiments on reddit data.
VADER Sentiment Analysis :
(Valence Aware Dictionary and sentiment Reasoner) NLTK built-in Vader sentiment analyzer Words that use positive and negative vocabulary will rank a piece of text as positive, negative or neutral. We can create one by first Emotional Strength Analyzer (SIA)To classify our title to take advantage of this tool, then we will use that polarity_scores Ways to get emotions.
Acctually,
Vader is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive, negative or neutral.

# characteristics of a Positive, Negative And Neutral comment,
sentiment analysis or sentiment classification fall into the broad category of text classification tasks where you are supplied with a phrase, or a list of phrases and your classifier is supposed to tell if the sentiment behind that is positive, negative or neutral.
Consider the following phrases:
1. "i love this movie."
2. "this movie is not a great movie."
3. "This is a movie."
The phrases correspond to short movie reviews, and each one of them conveys different sentiments.
For example,
the first phrase denotes positive sentiment about the film,
while the second one treats the movie as not so great (negative sentiment).
Take a look at the third one more closely. There is no such word in that phrase which can tell you about anything

# A Confusion Matrix
A confusion matrix can help understand the performance of the models.
We done The metrics and the confusion matrix for each model (SVM, the other model, and baseline), to compare their performance more appropriately;
Evaluation: To evaluate the model’s accuracy, a confusion matrix of the model is plotted using scikit-learn
The confusion matrix tabulates the number of correct predictions versus the number of incorrect predictions for each class, so it becomes easier to see which classes are the least accurately predicted for a given classifier
Transforming words to features: To transform the text into features, the first step is to use scikit-learn’s CountVectorizer.
This converts the entire corpus (i.e. all sentences) of our training data into a matrix of token counts.
Tokens (words, punctuation symbols, etc.) are created using NLTK’s tokenizer and commonly-used stop words like “a”, “an”, “the” are removed, because they do not add much value to the sentiment scoring.
in the Next, the count matrix is converted to a TF-IDF (Term-frequency Inverse document frequency) representation. From the scikit-learn documentation #Confusion Matrix is used to understand the trained classifier behavior over the test dataset or validate dataset.
First, We should split Our Data to training and testing, we use sklearn
With Using Sikite Lern Library, we Dividing test command and Train,
We consider 33% of the data as a test And the other as the Train.
We will use From this data for training And model layout.
One way to convert vector to natural language processing is TF-IDF, ( In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency,)
is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus Which is calculated as follows for each comment based on its words.
TF-IDF Features Image for post TF-IDF stands for Term Frequency-Inverse Document Frequency, and the TF-IDF weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.

# Accuracy
represents the number of correctly classified data instances over the total number of data instances.

# Recall (Sensitivity) 
Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes

# Precision 
Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.

# F1 score 
F1 Score is the weighted average of Precision and Recall.

# Decision Tree Algorithm
Once the data has been divided into the training and testing sets,
the next step is to train the decision tree algorithm on this data and make predictions. we will use the DecisionTreeClassifier.
The fit method of this class is called to train the algorithm on the training data, which is passed as parameter to the fit method.
Evaluating the Algorithm At this point we have trained our algorithm and made some predictions. Now we'll see how accurate our algorithm is. For classification tasks some commonly used metrics are confusion matrix, precision, recall, and F1 score.
Scikit-Learn's metrics library contains the classification_report and confusion_matrix methods that can be used to calculate these metrics for us: #training with using entropy (information gain) for decision tree What is Entropy? In the most layman terms, Entropy is nothing but the measure of disorder.

# Dummy Classification
The dummy classifier gives you a measure of “baseline” performance — i.e. the success rate one should expect to achieve even if simply guessing.
This classifier is useful as a simple baseline to compare with other (real) classifiers.
A dummy classifier is a type of classifier which does not generate any insight about the data and classifies the given data using only simple rules
It is used only as a simple baseline for the other classifiers
i.e. any other classifier is expected to perform better on the given dataset. It is especially useful for datasets where are sure of a class imbalance.
Actually, dummy classifier completely ignores the input data. In case of 'most frequent' method, it checks the occurrence of most frequent label

# LinearSVC
Now we used in this project is a support vector machine,
This model created with using the Sklearn Library,
And will be taught with the training data created in the following steps.
In the figure , you can see the code for creating, Training , and evaluating the model.
Here, 25% of the training data is considered as evaluation data for the selection of hyperparameters.
The value of c, which is the input of the backup vector machine model, is considered to be 0.5 , Evaluate for best performance on data.

# Conclusion:
In this project used data from Reddit's comments to identify polarity, such as positive, negative, or neutral.
First, the following pre-processing steps were performed on each comment:
#Preprocessing :
in this part Text and query used or not using redundant words, Taken And we do the following operations
1.with Using the function remove() And regular phrases And related Attempts have been made to remove the tags HTML
2.A list of Appastrofs has been compiled And is used to convert to full phrases,, For example they've becomes to they have
3.Some of them are of the Aaaa type Which represent certain emoticons Have been removed,, For example <3 which indicates the heart
4.Words in which some letters (feelings) are repeated Correct it,, For example happppppppy to happy
5.remove URLs
6.Excess marks are removed like , . ! ? @ ....
7.single-character characters have been removed
8.lemmatize
9.Stemming
10.Distance lines Repetitious has been removed
Then
consider With a part of the data as training data , And part as evaluation data, And part of as test data.
The results showed that the volume of training data has an effect on accuracy.
and with using scikit-learn library, we used DecisionTreeClassifier, DummyClassifier, LinearSVC,
And with using confusion matrixcfor each model, we understand the performance of the models.
Finally on the evaluation part, compute the recall, precision, and f1-score per each class.
And We Indicated Some characteristics of a Positive, Negative And Neutral comment,


