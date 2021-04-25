import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import tqdm
import re
import spacy


def tokenize_sentences(sentences, words_dict):
    # nlp = spacy.load('en', add_vectors=False, disable=['tagger', 'parser', 'ner'])
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        sentence = clean_text(sentence)
        sentence = text_parse(sentence)
        # tokens = text_to_wordlist(sentence)

        # tokens = nlp(sentence)
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            # word = word.text
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


# cleaning function
def clean_text(text, remove_stopwords=False, stem_words=False):
    """function to clean text data"""
    if remove_stopwords:
        text = text.lower().split()
        stops = set(stopwords.words('english'))
        text = [w for w in text if not w in stops]
        text = ' '.join(text)

    # Replace ips
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"&lt;3", "good", text)
    text = re.sub(r":d", "good", text)
    text = re.sub(r":dd", "good", text)
    text = re.sub(r":p", "good", text)
    text = re.sub(r"8\)", "good", text)
    text = re.sub(r":-\)", "good", text)
    text = re.sub(r":\)", "good", text)
    text = re.sub(r";\)", "good", text)
    text = re.sub(r"\(-:", "good", text)
    text = re.sub(r"yay!", "good", text)
    text = re.sub(r"yay", "good", text)
    text = re.sub(r"yaay", "good", text)
    text = re.sub(r"yaaay", "good", text)
    text = re.sub(r"yaaay", "good", text)
    text = re.sub(r"yaaaay", "good", text)
    text = re.sub(r"yaaaaay", "good", text)
    text = re.sub(r":/", "bad", text)
    text = re.sub(r":&gt;", "sad", text)
    text = re.sub(r":'\)", "sad", text)
    text = re.sub(r":-\(", "bad", text)
    text = re.sub(r":\(", "bad", text)
    text = re.sub(r":s", "bad", text)
    text = re.sub(r":-s", "bad", text)
    text = re.sub(r"&lt;3", "heart", text)
    text = re.sub(r":d", "smile", text)
    text = re.sub(r":p", "smile", text)
    text = re.sub(r":dd", "good", text)
    text = re.sub(r"\br\b", "are", text)
    text = re.sub(r"\bu\b", "you", text)
    text = re.sub(r"\bhaha\b", "ha", text)
    text = re.sub(r"\bhahaha\b", "ha", text)
    text = re.sub(r"\bdon't\b", "do not", text)
    text = re.sub(r"\bdoesn't\b", "does not", text)
    text = re.sub(r"\bdidn't\b", "did not", text)
    text = re.sub(r"\bhasn't\b", "has not", text)
    text = re.sub(r"\bhaven't\b", "have not", text)
    text = re.sub(r"\bhadn't\b", "good", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bwouldn't\b", "would not", text)
    text = re.sub(r"haha", "ha", text)
    text = re.sub(r"hahaha", "ha", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hadn't", "had not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"cannot", "can not", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"weren't", "were not", text)

    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    #     text = re.sub(r"\s{2,}", " ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\d+', ' ', text)
    text = re.sub('\s+', ' ', text)

    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('0', ' zero ')
    text = text.replace('1', ' one ')
    text = text.replace('2', ' two ')
    text = text.replace('3', ' three ')
    text = text.replace('4', ' four ')
    text = text.replace('5', ' five ')
    text = text.replace('6', ' six ')
    text = text.replace('7', ' seven ')
    text = text.replace('8', ' eight ')
    text = text.replace('9', ' nine ')

    text = text.strip(' ').lower()
    #     text = text.strip(' ')
    #     return text.split()

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text


def get_bad_word_dict():
    lines = open(
        './Kaggle/Toxic Comment Classification Challenge/notebooks/toxic-master/toxic/badwords.list').readlines()
    lines = [l.lower().strip('\n') for l in lines]
    lines = [l.split(',') for l in lines]
    bad_dict = {}
    for v in lines:
        if len(v) == 2:
            bad_dict[v[0]] = v[1]
    return bad_dict


bad_word_dict = get_bad_word_dict()


def text_parse(text, remove_stopwords=False, stem_words=False):
    wiki_reg = r'https?://en.wikipedia.org/[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    url_reg = r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    url_reg2 = r'www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    ip_reg = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    WIKI_LINK = ' WIKILINKREPLACER '
    URL_LINK = ' URLLINKREPLACER '
    IP_LINK = ' IPLINKREPLACER '
    # clear link
    # replace endline with '. '
    endline = re.compile(r'.?\n', re.IGNORECASE)
    text = endline.sub('. ', text)

    c = re.findall(wiki_reg, text)
    for u in c:
        text = text.replace(u, WIKI_LINK)
    c = re.findall(url_reg, text)
    for u in c:
        text = text.replace(u, URL_LINK)
    c = re.findall(url_reg2, text)
    for u in c:
        text = text.replace(u, URL_LINK)
    c = re.findall(ip_reg, text)
    for u in c:
        text = text.replace(u, IP_LINK)

    # bad_word_dict = get_bad_word_dict()
    # Regex to remove all Non-Alpha Numeric and space
    special_character_removal = re.compile(r'[^A-Za-z\d!?*\'.,; ]', re.IGNORECASE)
    # regex to replace all numerics
    replace_numbers = re.compile(r'\b\d+\b', re.IGNORECASE)
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Remove Special Characters
    text = special_character_removal.sub(' ', text)
    for k, v in bad_word_dict.items():
        # bad_reg = re.compile('[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ]'+ re.escape(k) +'[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ]')
        bad_reg = re.compile('[\W]?' + re.escape(k) + '[\W]|[\W]' + re.escape(k) + '[\W]?')
        text = bad_reg.sub(' ' + v + ' ', text)
        '''
        bad_reg = re.compile('[\W]'+ re.escape(k) +'[\W]?')
        text = bad_reg.sub(' '+ v, text)
        bad_reg = re.compile('[\W]?'+ re.escape(k) +'[\W]')
        text = bad_reg.sub(v + ' ', text)
        '''

    # Replace Numbers
    text = replace_numbers.sub('NUMBERREPLACER', text)
    text = text.split()
    text = " ".join(text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # rake parsing
    # text = rake_parse(text)
    return text


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    wiki_reg = r'https?://en.wikipedia.org/[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    url_reg = r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    ip_reg = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    WIKI_LINK = ' WIKI_LINK '
    URL_LINK = ' URL_LINK '
    IP_LINK = ' IP_LINK '
    # clear link
    c = re.findall(wiki_reg, text)
    for u in c:
        text = text.replace(u, WIKI_LINK)
    c = re.findall(url_reg, text)
    for u in c:
        text = text.replace(u, WIKI_LINK)
    c = re.findall(wiki_reg, text)
    for u in c:
        text = text.replace(u, URL_LINK)
    c = re.findall(ip_reg, text)

    # Regex to remove all Non-Alpha Numeric and space
    special_character_removal = re.compile(r'[^A-Za-z\d!?*\' ]', re.IGNORECASE)
    # regex to replace all numerics
    replace_numbers = re.compile(r'\d+', re.IGNORECASE)

    # text = text.lower().split()
    text = text.split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)
    # Remove Special Characters
    text = special_character_removal.sub('', text)
    # Replace Numbers
    text = replace_numbers.sub('NUMBERREPLACER', text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # Return a list of words
    return text


