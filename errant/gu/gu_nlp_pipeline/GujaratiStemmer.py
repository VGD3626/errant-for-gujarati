import re
stopwords, suffixes, prefixes = [], [], []

# You may need to add/remove suffixes/prefixes according to the corpora
suffixes = ['નાં','ના','ની','નો','નું','ને','થી','માં','એ','ીએ','ઓ','ે','તા','તી','વા','મા','વું','વુ','ો','માંથી','શો','ીશ','ીશું','શે',
			'તો','તું','તાં','્યો','યો','યાં','્યું','યું','ોઈશ', 'ોઈશું', '્યા','યા','્યાં','સ્વી','રે','ં','મ્','મ્','ી','કો',
      'ેલ', 'ેલો', 'ેલા', 'ેલું', 'ેલી', 'ણે', 'ણા', 'ણું', 'ણો', 'ણી'
      ]
prefixes = [] #['અ']

def SentenceTokenizer(data):
    data = data.strip()
    data = re.sub(r'([.!?])', r'\1 ', data)
    data = re.split(r'  ',data)
    if not data[-1]:
        del(data[-1])
    return data

def WordTokenizer(data, keep_stopwords = True):
        
        data = re.sub(r'([.,\'\\"!?%#@*<>|\+\-\(\)])', r' \1', data)
        data = re.sub(r'[।।(૧૨૩૪૫૬૭૮૯)*।।]', '  ', data)
        data = re.sub(r"   ", '', data)
        data = re.sub(r'…', " ", data)
        data = re.split(r'[ -]',data)
        words = []
        
        if not keep_stopwords:
            for word in data:
                if word not in stopwords:
                    words.append(word)
            # spaces=(len(words)-1)*[True]+[False]
            return words

        for i in data:
            if i:
                words.append(i)
        # spaces=(len(words)-1)*[True]+[False]
        return words  

class Preprocessor():
    def __init__(self):
        self.suffixes = []
        pass

    def compulsory_preprocessing(self, text):
        '''This is a function to preprocess the text and make the necessary changes which are compulsory for any type of Gujarati NLP task'''
        text = re.sub(r'\u200b', '', text)
        text = re.sub(r'\ufeff', "", text)
        text = re.sub(r'…', " ", text)
        text = re.sub(r'  ', ' ', text)
        text = re.sub(r'”“', '', text)
        text = WordTokenizer(text)
        for i in range(len(text)):
            text[i] = text[i].rstrip(':')
        return ' '.join(text)

    def remove_tek(self, text, tek_string):
        '''
        Tek is the Gujarati word for the initial line of the poem. Whenever, one stanza of any poem is sung, the initial line of the poem is sung once again before starting the
        next stanza. This is called as singing a "Tek". Written poems mention the tek string too many a times. This will cause a problem of redundancy. Hence, removing it is
        necessary.
        '''
        if str(type(tek_string))=="<class 'NoneType'>" or not tek_string:
            raise TypeError('tek_string needs to be a valid string')
        if str(type(text))=="<class 'list'>":
            for i in range(len(text)):
                text[i] = text[i].rstrip(tek_string)
        elif str(type(text))=="<class 'str'>":
            text = text.rstrip(tek_string)
        else:
            raise TypeError("Argument 'text' must be either a str or list")
        return text

    def poetic_preprocessing(self, text, remove_tek=False, tek_string=None):
        '''This function is only required when dealing with poetic corpora. Make sure to use this function along with the compulsory preprocessing to have decently accurate results with poetic corpora'''
        text = re.sub(r'।','.',text)
        text = re.sub(' ।।[૧૨૩૪૫૬૭૮૯૦]।।', '.', text)
        if remove_tek:
            text = self.remove_tek(text, tek_string)
        tokens = WordTokenizer(text, corpus='poetry', keep_punctuations=False)

        for i in range(len(tokens)):
            # Rule 1
            if tokens[i].endswith('જી'):
                tokens[i] = tokens[i].strip('જી')
            # Rule 2
            if tokens[i].endswith('ૈ'):
                tokens[i] = tokens[i].strip('ૈ')+'ે'
            # Rule 3
            index = tokens[i].find('ર')
            if index == -1:
                pass
            elif index<len(tokens[i])-1 and tokens[i][index-1]=='િ':
                tokens[i] = re.sub('િર', 'ૃ', tokens[i])

        return ' '.join(tokens)

class Stemmer():
	def __init__(self):
		self.suffixes = suffixes
		self.prefixes = prefixes

	def add_suffix(self, suffix):
		self.suffixes.append(suffix)

	def add_prefix(self, prefix):
		self.prefixes.append(prefix)

	def delete_suffix(self, suffix):
		try:
			del(self.suffixes[self.suffixes.index(suffix)])
		except IndexError:
			print('{} not present in suffixes'.format(suffix))

	def delete_prefix(self, prefix):
		try:
			del(self.prefixes[self.prefixes.index(prefix)])
		except IndexError:
			print("{} not present in prefixes".format(prefix))


	def stem_word(self, sentence, corpus):
		word_list = sentence.strip('\u200b').split(' ')
		if not word_list[-1]:
			del(word_list[-1])
		return_list = []
		suffix_list = []
		puctuations = ('.',',','!','?','"',"'",'%','#','@','&','…','“', '”', '’', '‘', ':', ';')
		for word in word_list:
			a = word
			removed_suffix = None
			if word.endswith(puctuations):
				a = word[:-1]
			if a in stopwords:
					return_list.append(a)
					suffix_list.append(None)
					continue
			for suffix in suffixes:
				if a.endswith(suffix):
					a = a[:-len(suffix)]
					removed_suffix = suffix
					break
			for prefix in prefixes:
				if a.startswith(prefix):
					a = a[len(prefix):]
					break
			if word.endswith(puctuations):
				a += str(word[-1])
			return_list.append(a)
			suffix_list.append(removed_suffix)
		return_sentence = " ".join(return_list)
		return return_sentence

	def stem(self, text, corpus='prose', remove_tek=False, tek_string=None):
		preprocessor = Preprocessor()
		text = preprocessor.compulsory_preprocessing(text)
		if corpus == 'poetry':
			text = preprocessor.poetic_preprocessing(text, remove_tek=remove_tek, tek_string=tek_string)
		elif corpus == 'prose':
			pass
		else:
			raise ValueError("Unnrecognized argument 'corpus'. Should be either 'prose' or 'poetry'")
		l = SentenceTokenizer(text)
		if len(l)==1:
			sentence = l[0]
			return self.stem_word(sentence, corpus=corpus)
		else:
			a = []
			for sentence in l:
				a.append(self.stem(sentence))
			return a
		