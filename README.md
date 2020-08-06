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

![Untitled](https://user-images.githubusercontent.com/38555065/89587993-98487200-d7f7-11ea-863d-1111b198cfa9.png)


# Stemming and Lemmatization are Text Normalization (or sometimes called Word Normalization)
techniques in the field of Natural Language Processing that are used to prepare text, words, and documents for further processing.

![Untitled1](https://user-images.githubusercontent.com/38555065/89588186-fe34f980-d7f7-11ea-89b6-ba858113571a.png)

# Stemming
Stemming refers to reducing a word to its root form.
While performing natural language processing tasks, you will encounter various scenarios where you find different words with the same root.
For instance, compute, computer, computing, computed, etc.
You may want to reduce the words to their root form for the sake of uniformity.

# Lemmatization
Lemmatization is the process of converting a word to its base form.

# reading the Jason file and calling its preprocessor 
It also clears the input data function And tags In the form of a label :
1 indicates positive , And the 0 tag is a sign of neutrality, And label -1 indicates that the comment is negative.

![Untitled2](https://user-images.githubusercontent.com/38555065/89588357-579d2880-d7f8-11ea-9219-1e89d16d64df.png)

![Untitled13](https://user-images.githubusercontent.com/38555065/89589922-9e405200-d7fb-11ea-8679-a8ca9b8ce7b7.png)


# Clean Word for labeling the comments i use SentimentIntensityAnalysis library from nltk. 

![Untitled4](https://user-images.githubusercontent.com/38555065/89588592-d003e980-d7f8-11ea-96be-22138a5ed1b6.png)

# Remove punctuation
The following code removes this set of symbols [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:
Remove punctuation and numbers in text,


![Untitled3](https://user-images.githubusercontent.com/38555065/89588502-9e8b1e00-d7f8-11ea-8a4a-1382bc6905d1.png)


# What is NLTK?
NLTK stands for Natural Language Toolkit. This toolkit is one of the most powerful NLP libraries which contains packages to make machines understand human language and reply to it with an appropriate response.
Tokenization, Stemming, Lemmatization, Punctuation, Character count, word count are some of these packages.
With Using the library nltk, we downloading List of stopwords and Markings And we download network words.
The project uses the nltk library It is received with the following lines of this library.

![Untitled5](https://user-images.githubusercontent.com/38555065/89588798-3426ad80-d7f9-11ea-9331-ee599536196f.png)




# NLTK -> Vader
(SentimentIntensityAnalyzer) results with a Machine Learning trained classifier for prediction of Sentiments on reddit data.
VADER Sentiment Analysis :
(Valence Aware Dictionary and sentiment Reasoner) NLTK built-in Vader sentiment analyzer Words that use positive and negative vocabulary will rank a piece of text as positive, negative or neutral. We can create one by first Emotional Strength Analyzer (SIA)To classify our title to take advantage of this tool, then we will use that polarity_scores Ways to get emotions.
Acctually,
Vader is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive, negative or neutral.


![Untitled6](https://user-images.githubusercontent.com/38555065/89588897-6f28e100-d7f9-11ea-92e9-211cc4d5375c.png)



# With import Matplotlib library 
we show the plot related to the division of comments,
As we can see, most of the comments belong to neutral and show with 0,
The result is a clean data file for further processing.

![Untitled7](https://user-images.githubusercontent.com/38555065/89589135-e1012a80-d7f9-11ea-96a3-0f3e8e556eac.png)
![Untitled8](https://user-images.githubusercontent.com/38555065/89589336-46edb200-d7fa-11ea-8418-26b3b0452a3f.png)




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

![Untitled12](https://user-images.githubusercontent.com/38555065/89589773-28d48180-d7fb-11ea-8fce-e62accdbb6b9.png)

![Untitled9](https://user-images.githubusercontent.com/38555065/89589495-903e0180-d7fa-11ea-8c3a-02154960bcb9.png)
![Untitled10](https://user-images.githubusercontent.com/38555065/89589584-bbc0ec00-d7fa-11ea-8a8d-b12850798d3e.png)
![Untitled11](https://user-images.githubusercontent.com/38555065/89589673-ed39b780-d7fa-11ea-9016-ac847feb9e33.png)



# Accuracy
represents the number of correctly classified data instances over the total number of data instances.

![Untitled20](https://user-images.githubusercontent.com/38555065/89590920-c7fa7880-d7fd-11ea-98a9-29e64d4615b7.png)


# Recall (Sensitivity) 
Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes

![Untitled21](https://user-images.githubusercontent.com/38555065/89590942-d3e63a80-d7fd-11ea-8963-d3030d2445de.png)


# Precision 
Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.

![Untitled22](https://user-images.githubusercontent.com/38555065/89590959-dcd70c00-d7fd-11ea-9220-1c6b8ddfd1c4.png)

# F1 score 
F1 Score is the weighted average of Precision and Recall.

![Untitled23](https://user-images.githubusercontent.com/38555065/89590970-e5c7dd80-d7fd-11ea-9d3a-fe9e3ca9858e.png)



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

![Untitled14](https://user-images.githubusercontent.com/38555065/89590100-faa37180-d7fb-11ea-9c62-c777ffdbcc83.png)

![Untitled15](https://user-images.githubusercontent.com/38555065/89590181-1eff4e00-d7fc-11ea-9ac7-17a4059de733.png)



# Decision Tree Algorithm
Once the data has been divided into the training and testing sets,
the next step is to train the decision tree algorithm on this data and make predictions. we will use the DecisionTreeClassifier.
The fit method of this class is called to train the algorithm on the training data, which is passed as parameter to the fit method.
Evaluating the Algorithm At this point we have trained our algorithm and made some predictions. Now we'll see how accurate our algorithm is. For classification tasks some commonly used metrics are confusion matrix, precision, recall, and F1 score.
Scikit-Learn's metrics library contains the classification_report and confusion_matrix methods that can be used to calculate these metrics for us: #training with using entropy (information gain) for decision tree What is Entropy? In the most layman terms, Entropy is nothing but the measure of disorder.

![Untitled16](https://user-images.githubusercontent.com/38555065/89590272-50781980-d7fc-11ea-9bd0-35cab79ce131.png)

![Untitled17](https://user-images.githubusercontent.com/38555065/89590315-72719c00-d7fc-11ea-8929-ad687986cf5e.png)


# Dummy Classification
The dummy classifier gives you a measure of “baseline” performance — i.e. the success rate one should expect to achieve even if simply guessing.
This classifier is useful as a simple baseline to compare with other (real) classifiers.
A dummy classifier is a type of classifier which does not generate any insight about the data and classifies the given data using only simple rules
It is used only as a simple baseline for the other classifiers
i.e. any other classifier is expected to perform better on the given dataset. It is especially useful for datasets where are sure of a class imbalance.
Actually, dummy classifier completely ignores the input data. In case of 'most frequent' method, it checks the occurrence of most frequent label

![Untitled18](https://user-images.githubusercontent.com/38555065/89590492-e3b14f00-d7fc-11ea-991b-8ab9381ad617.png)



# LinearSVC
Now we used in this project is a support vector machine,
This model created with using the Sklearn Library,
And will be taught with the training data created in the following steps.
In the figure , you can see the code for creating, Training , and evaluating the model.
Here, 25% of the training data is considered as evaluation data for the selection of hyperparameters.
The value of c, which is the input of the backup vector machine model, is considered to be 0.5 , Evaluate for best performance on data.

![Untitled19](https://user-images.githubusercontent.com/38555065/89590575-10656680-d7fd-11ea-9574-a18d201bc01b.png)


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


