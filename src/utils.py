
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tokenizers, Stemmers and Lemmatizers
import nltk
from nltk.corpus import stopwords
import spacy
# Download resources
nltk.download("stopwords")
stopwords = set(stopwords.words("english"))
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# precision, recall, f1-score,
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
    
class Utils:
    
        # remove special characters
    @staticmethod
    def clean_sentence(text):
        #Removing numerical values, Removing Digits and words containing digits
        text= re.sub('\w*\d\w*','', text)
        #Removing punctations
        text= re.sub('[%s]' % re.escape(string.punctuation), '', text)
        #Removing Extra Spaces
        text =re.sub(' +', ' ',text)
        # remove stock market tickers like $GE
        text = re.sub(r'\$\w*', '',text)
        # remove old style retweet text "RT"
        text = re.sub(r'^RT[\s]+', '',text)
        # remove hyperlinks
        text = re.sub(r'https?:\/\/.*[\r\n]*', '',text)
        # remove hashtags
        # only removing the hash # sign from the word
        text = re.sub(r'#', '',text)
        text = text.lower()

        return text
    
    @staticmethod
    def encode_text(text):
        sequence = tokenizer.texts_to_sequences([text])
        padded_test_sequence = pad_sequences(sequence, maxlen=30)
        print(padded_test_sequence)
        return padded_test_sequence

    @staticmethod
    # remove special characters
    def remove_special_characters(texts):
        #Removing numerical values, Removing Digits and words containing digits
        l_texts= texts.apply(lambda x: re.sub('\w*\d\w*','', x))
        #Removing punctations
        l_texts= l_texts.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
        #Removing Extra Spaces
        l_texts = l_texts.apply(lambda x: re.sub(' +', ' ',x))
        # remove stock market tickers like $GE
        l_texts = l_texts.apply(lambda x: re.sub(r'\$\w*', '',x))
        # remove old style retweet text "RT"
        l_texts = l_texts.apply(lambda x: re.sub(r'^RT[\s]+', '',x))
        # remove hyperlinks
        l_texts = l_texts.apply(lambda x: re.sub(r'https?:\/\/.*[\r\n]*', '',x))
        # remove hashtags
        # only removing the hash # sign from the word
        l_texts = l_texts.apply(lambda x: re.sub(r'#', '',x))

        return l_texts
    
    @staticmethod
    def tokenize_stopwords_lemmatize(texts, allowed_postags=['NOUN','ADJ','ADV']):
        tokenized_docs = texts.apply(lambda x: ' '.join([token.lemma_.lower() for token in list(nlp(x)) if token.is_alpha and not              token.is_stop]))
        return tokenized_docs
    
    @staticmethod
    def analyse_performance_model(model,X,y,title_dataset):
        y_pred_proba = model.predict(X)
        y_pred = np.where(y_pred_proba> 0.5, 1, 0)
        cf = confusion_matrix(y, y_pred)
        plt.figure()
        Utils.make_confusion_matrix(cf, categories=['NEGATIVE', 'POSITIVE'], title="Performance du modèle sur le "+title_dataset)
        plt.figure()
        Utils.plot_roc_curve(y_pred_proba,y,title='Courbe ROC sur le ' + title_dataset)

    @staticmethod    
    def plot_roc_curve(y_pred_proba,y_true,title=None):
        #define metrics
        auc = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_true,  y_pred_proba)
        #create ROC curve
        #create ROC curve
        plt.plot(fpr,tpr,label="AUC="+str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.title(title)

    @staticmethod 
    def make_confusion_matrix(cf,
                              group_names=None,
                              categories='auto',
                              count=True,
                              percent=True,
                              cbar=True,
                              xyticks=True,
                              xyplotlabels=True,
                              sum_stats=True,
                              figsize=None,
                              cmap='Blues',
                              title=None):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        '''


        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names)==cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cf) / float(np.sum(cf))

            #if it is a binary confusion matrix, show some more stats
            if len(cf)==2:
                #Metrics for Binary Confusion Matrices
                precision = cf[1,1] / sum(cf[:,1])
                recall    = cf[1,1] / sum(cf[1,:])
                f1_score  = 2*precision*recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy,precision,recall,f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""


        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False


        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)
