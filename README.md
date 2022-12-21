# Detecting-Depression-NLP
WiDS Project

<h2><b>Week 1</b></h2>
  <h3>Dataset Used</h3> <a href='https://www.kaggle.com/datasets/ywang311/twitter-sentiment'></a>
  This dataset has file Sentiment Analysis Dataset 2.csv having following columns:<br>
  <ol>
    <li>ItemID : Unique ID for tweets
    <li>Sentiment : 1 if tweet is of depression sentiment, 0 if not
    <li>SentimentSource : Source of tweet extraction
    <li>SentimentText : Raw text of tweet
  </ol>

  <h2>Text Pre Processing</h2>
  <ol>
    <li>Drop unnecessary columns (ID, Source)
    <li>Removing non-useful characters (punctuation,stopwords)
    <li>Perform tokenization
    <li>Perform stemming and lemmatization
  </ol>
  <h2>Exploratory Data Analysis </h2>
