<h1>
    During pre-processing of data, some trends were observed which are listed below
</h1>

<ul>
    <li>Tweets labelled as '1' are said to be "depressing" in nature and '0' otherwise.
    <li>The dataset has 1.5m tweets, to speed up processing, we drop non-relevant columns (ItemID and Source of the Tweet).
    <li>The dataset has no null values and is has nearly zero skew between classes. 
    <li>Tweets also have punctuation, which will not be useful for further analysis and hence we drop punctuation (, . - ! ? : ; ' " ")
    <li>Drop URL's, to avoid training the model on words present in the url.
    <li>All tweets are converted to lowercase
</ul>
<p>Fun fact : The difference between the ascii values of a uppercase alphabet and a lowercase is <b> 32</b>.</p>

<h3>Skew or no Skew</h3>

<p>
    We have nearly equal number of datapoints from both classes therby showing that the dataset has zero-to-no skew.
</p>
![Bar plot showing distribution of tweets, 1 represents tweet that is classified as depressed, 0 represents otherwise](https://user-images.githubusercontent.com/87320561/209464632-df7ad6de-952a-4f07-bc3a-1e9588bdbb7e.png)

```python
#Code for above visualization
#requires 3.6.2 version of matplotlib
ax=df['Sentiment'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))
ax.bar_label(ax.containers[-1])
ax.margins(y=0.1)
plt.show()
```
<h3>What are stopwords ?</h3> 

<p>
Common words that provide little to no meaning to a sentence, we drop them to focus more on "rare"-words.<br>
Stop words occur in abundance, hence providing little to no unique information that can be used for classification or clustering.<br>
By default, NLTK (Natural Language Toolkit) includes a list of 40 stop words, including: “a”, “an”, “the”, “of”, “in”, etc. <br>
<a href="https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html">Read more about stopwords :</a>
</p>


<h3>Comparing length of tweets before and after removing stopwords</h3>

<ul>
<li><p><big>Before removing stopwords from tweets :</big></p>
![KDE plot showing length of tweets before any pre-processing](https://user-images.githubusercontent.com/87320561/209464659-132c86dd-f1fd-4661-b025-f9bd2744cfd8.png)

<p>Code for KDE plots of length of tweets before removing stopwords</p>

```python
#import seaborn as sns
#import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 3))
transparency=0.6
sns.kdeplot(df.loc[(df['Sentiment']==0), 
            'length'],
            color='crimson', label='Not depressed', ax=ax)

sns.kdeplot(df.loc[(df['Sentiment']==1), 
            'length'],
            color='blue', label='Depressed', ax=ax)
ax.legend()
plt.tight_layout()
plt.title('Before removing StopWords')
plt.show()
```


<li><p><big>After pre-processing of tweets:</big></p>

![KDE plot showing length of tweets before any pre-processing](https://user-images.githubusercontent.com/87320561/209464671-8c59d769-0151-463e-afdd-8c2241e2e48a.png)

<p>Code for KDE plots of length of tweets after removing stopwords </p>

```python
#import seaborn as sns
#import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 3))
transparency=0.6
sns.kdeplot(df.loc[(df['Sentiment']==0), 
            'len after'],
            color='crimson', label='Not depressed', ax=ax)

sns.kdeplot(df.loc[(df['Sentiment']==1), 
            'len after'],
            color='blue', label='Depressed', ax=ax)
ax.legend()
plt.tight_layout()
plt.title('After Removing Stopwords')
plt.show()
```
</ul>
<br>
<h4 style="font-weight: normal;">No trend-difference is seen in length of tweets, based on classes, which indicates this might not be a good classification feature.</h4>

<h3>Word Clouds showing most frequent words</h3>
<ol>
<li>
<p>For complete dataset : </p> 
![Wordcloud for complete dataset, requires 98s to run](https://user-images.githubusercontent.com/87320561/209464695-a2a8cdcd-7f86-467b-8fe1-c926f602e67c.png)

<li>
<p>For only not-depressing tweets :</p>
![Wordcloud for tweets with sentiment0, requires 50-55s to run](https://user-images.githubusercontent.com/87320561/209464725-2bc43581-e447-4d8e-a6e8-1e44e7364147.png)
<li>
<p>For only non-depressing tweets :</p>
![Wordcloud for tweets with sentiment1, requires 50-55s to run](https://user-images.githubusercontent.com/87320561/209464733-18f27807-2859-4917-8d27-123ef4c87aeb.png)
