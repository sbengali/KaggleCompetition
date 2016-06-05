import nltk

# Function to remove unnecessary parts of input recipe data
def filterstring(str):
    # Remove copyright, registered, trademark, acute a, circumflex a, manada a, section, cent, latin c with cedilla
    symbols_list = [u"\u00a9", u"\u00ae", u"\u2122", u"\u00e1", u"\u00e2", u"\u00e3", u"\u00a7", u"\u00a2", u"\u00e7"]
    # Convert string to lower case
    str = str.lower()
    # Remove any symbols
    for s in symbols_list:
        new_string = str.replace(s, "")
        str = new_string

    tokens = nltk.word_tokenize(str)
    tags = nltk.pos_tag(tokens)
    force_tags = {'flour': 'NN', 'garlic': 'NN'}
    new_tags = [(word, force_tags.get(word, tag)) for word, tag in tags]
    str = [t for t in new_tags if t[1] == "NN" or t[1] == "NNS" or t[1] == "VBG"]
    str = [word for (word, tag) in str]
    # Remove whitespace and non-alpha characters
    str = "".join([i for i in str if i.isalpha()])
    # Return the processed string
    return str