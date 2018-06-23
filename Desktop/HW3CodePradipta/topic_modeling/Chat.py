import pyLDAvis.gensim
import pandas as pd;
import numpy as np;
from nltk.corpus import stopwords;
from gensim.models import ldamodel,LsiModel,CoherenceModel
import gensim.corpora;
import pickle;
import matplotlib.pyplot as plt

#chat = pd.read_csv('Feb19_GroupB.csv', header=None, names=['Text','Name','TimeStamp'])
chat = pd.read_csv('Mar11_GroupB.csv', header=None, names=['Name','Text','TimeStamp'])

data_text = chat[['Text']];
np.random.seed(1024);
data_text = data_text.iloc[np.random.choice(len(data_text), 500)];
data_text = data_text.astype('str');
for idx in range(len(data_text)):
    
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    data_text.iloc[idx]['Text'] = [word for word in data_text.iloc[idx]['Text'].split(' ') if word not in stopwords.words() and word.isalpha()];
    
    #print logs to monitor output
    if idx % 100 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)));
        
pickle.dump(data_text, open('data_text.dat', 'wb'))

train_chat = [value[0] for value in data_text.iloc[0:].values];
num_topics = 4;

id2word = gensim.corpora.Dictionary(train_chat)
corpus = [id2word.doc2bow(text) for text in train_chat]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

lsimodel = LsiModel(corpus=corpus, num_topics=num_topics, id2word=id2word)

def get_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);

get_topics(lda, num_topics)

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda, corpus, id2word)

lsitopics = [[word for word, prob in topic] for topicid, topic in lsimodel.show_topics(formatted=False)]
ldatopics = [[word for word, prob in topic] for topicid, topic in lda.show_topics(formatted=False)]

lsi_coherence = CoherenceModel(model=lsimodel,topics=lsitopics,dictionary=id2word, texts=train_chat,window_size=10).get_coherence()
lda_coherence = CoherenceModel(model=lda,topics=ldatopics,dictionary=id2word,texts=train_chat,window_size=10).get_coherence()

#lda_coherence =CoherenceModel(model=lsimodel, corpus=corpus, coherence='u_mass').get_coherence() 

def  evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    print(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')
    
    
evaluate_bar_graph([lsi_coherence,lda_coherence],
                   ['LSI','LDA'])
                   
