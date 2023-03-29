from functions import libraries_basic,libraries_neural,predict_tweet_sentiment
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive
os.chdir('MyDrive/Projects')
# !ls

file_path='Processed data.csv'
df=pd.read_csv(file_path,on_bad_lines='skip')

print("Shape of the dataframe is :",df.shape)

print("Entries with sentiment 1 :",df[df['Sentiment']==1].shape[0])
print("Entries with sentiment 0 :",df[df['Sentiment']==0].shape[0])

df.drop("Unnamed: 0",axis=1,inplace=True)

df['Score_sentiment']= np.where(df['Sentiment']== 1, "Positive", "Negative")

train_data, test_data = train_test_split(df, test_size=0.2,random_state=16)
print("Train Data size:", len(train_data))
print("Test Data size", len(test_data))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['SentimentText'])
word_index = tokenizer.word_index
print(word_index)

#takes nearly 30s to run


vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)
# Vocabulary Size : 6,66,541

# Tokens are converted into sequences then passed on to the pad_sequences() function

x_train = pad_sequences(tokenizer.texts_to_sequences(train_data['SentimentText']),maxlen = 30)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data['SentimentText']),maxlen = 30)

#takes nearly 30s to run

labels = ['Negative', 'Positive']

encoder = LabelEncoder()
encoder.fit(train_data.Sentiment.to_list())
y_train = encoder.transform(train_data.Sentiment.to_list())
y_test = encoder.transform(test_data.Sentiment.to_list())
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip


embeddings_index = {}
# opening the downloaded glove embeddings file
f = open('glove.6B.300d.txt')
for line in f:
    # For each line file, the words are split and stored in a list
    values = line.split()
    word = value = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' %len(embeddings_index))

# creating an matrix with zeroes of shape vocab x embedding dimension
embedding_matrix = np.zeros((vocab_size, 300))

# Iterate through word, index in the dictionary
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

embedding_layer = tf.keras.layers.Embedding(vocab_size,300,
                                            weights=[embedding_matrix],
                                            input_length=30,trainable=False)

# The Input layer 
sequence_input = Input(shape=(30,), dtype='int32')

# Inputs passed to the embedding layer
embedding_sequences = embedding_layer(sequence_input)

# dropout and conv layer 
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation='relu')(x)

# Passed on to the LSTM layer
x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)

# Passed on to activation layer to get final output
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)

model.compile(optimizer=Adam(), loss='binary_crossentropy',metrics=['accuracy'])

ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,min_lr = 0.01, monitor = 'val_loss',verbose = 1)

training = model.fit(x_train, y_train, 
                    batch_size=512, epochs=4,
                    validation_data=(x_test, y_test), 
                    callbacks=[ReduceLROnPlateau])

#17mins per epoch


scores = model.predict(x_test, verbose=1, batch_size=10000)
model_predictions = [predict_tweet_sentiment(score) for score in scores]

test_data['Score_sentiment']= np.where(test_data['Sentiment']== 1, "Positive", "Negative")
# test_data

"""Results"""

print(training.history)

fig=plt.figure()

#
ax1=fig.add_subplot(221)
ax1.plot(training.history['accuracy'],label='Train')
ax1.plot(training.history['val_accuracy'],label='Validation')

ax1.set(ylabel='Accuracy',xlabel='Epoch',title='Model Accuracy')
ax1.legend()
plt.show()

#
ax2=fig.add_subplot(222)
ax2.plot(training.history['loss'],label='Training Loss')
ax2.plot(training.history['val_loss'],label='Validation Loss')

ax2.set(xlabel="Epoch",ylabel='Loss',title='Loss vs Epoch')
ax2.legend()
plt.show()

print(classification_report(list(test_data.Score_sentiment), model_predictions))

pickle.dump(model, open('model.pkl', 'wb'))

# !ls # To verify pickle dump