from functions import import_libraries_clean,clean_words

import_libraries_clean()
# To get files from google drive
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive

os.chdir('MyDrive/Projects')
# !ls  #to check desired directory is opened

nltk.download('popular')
nltk.download('stopwords')

# Ignore FutureWarnings, warns about future package compatibility issues that may arise
warnings.simplefilter(action='ignore', category=FutureWarning)  

file_path='Sentiment Analysis Dataset 2.csv'
df=pd.read_csv(file_path,on_bad_lines='skip')

print("Shape of dataframe is :",df.shape)
# df.head() to observe the data

# Dropping unneccessary columns
df.drop(['SentimentSource','ItemID'],axis=1,inplace=True)

# Checking for null values ,returns boolean value
df.isnull().any() 

# Display count plot of the classes in column "Sentiment"
ax=df['Sentiment'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))
ax.bar_label(ax.containers[-1])
ax.margins(y=0.1)
plt.title('Count plot of tweets of diff classes')
plt.tight_layout()
plt.show()

clean_words()

#Adding length column in dataframe
df['length']=df["SentimentText"].str.len()

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
plt.savefig("KDEplots before removing stopwords")
plt.title('Before removing StopWords')
plt.show()

#Removing StopWords
stop_words = stopwords.words('english')
df['WithoutStopwords'] = df['SentimentText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
#takes around 1min to complete
df['len after']=df["WithoutStopwords"].str.len()


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
plt.savefig("KDEplots after removing stopwords")
plt.show()

# Removing very small tweets
df.drop(df[df.length<5].index , inplace=True)

# Weeding out tweets with length > 160
df.drop(labels=[413173,101506,286100,615914],axis=0,inplace=True)

# To save dataset for further analysis
df.to_csv('Processed data.csv')