import re
import string
from emoji import UNICODE_EMOJI
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer_back = WordNetLemmatizer()
allowed_letters = set(string.ascii_lowercase + " -") | set(UNICODE_EMOJI.keys())
allowed_letters2 = set(string.ascii_lowercase + " ") | set(UNICODE_EMOJI.keys())
emojis = set(UNICODE_EMOJI.keys())

memo = {}

# The lemmatizer converts all variants of the same word into the dictionary form,
# i.e. making all nouns into singular and all verbs into bare infinitive (etc.).
def lemmatizer(w):
    if w in memo:
        return memo[w]
    r = lemmatizer_back.lemmatize(w, get_wordnet_pos(w))
    if len(memo) < 50000:
        memo[w] = r
    return r

def preprocess(text):
    # Lowercasing the text
    text = text.lower()
    # Removing HTML-tags
    text = text.replace("<i>", " ")
    text = text.replace("\\n", " ")
    text = text.replace("<p>", " ")
    text = text.replace("</i>", " ")
    text = text.replace("</p>", " ")
    # Using a URL-token
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", "-url-", text)
    # Removing HTML special characters
    text = re.sub(r"\&.{0,4}\;", "", text)
    # Using an emoji-token
    text = "".join([" -emoji- " if s in emojis else s for s in text if s in allowed_letters])
    # Removing disallowed characters
    text = "".join([text[i] for i in range(len(text)) if (text[i] in allowed_letters2
        \ or (text[i+1:i+5] == "url-" or text[i+1:i+7] == "emoji-" 
        \ or text[i-4:i] == "-url" or text[i-6:i] == "-emoji"))])
    # Removing extra whitespace
    text = re.sub(r"\ +", " ", text)
    text = text.strip()
    # Lemmatizing the words
    text = " ".join([w if ("-emoji-" in w or "-url-" in w) else lemmatizer(w)
        \ for w in nltk.word_tokenize(text)])
    return text
