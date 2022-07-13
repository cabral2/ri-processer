from operator import contains
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

class Cleaner:
    def __init__(self, stop_words_file: str, language: str,
                 perform_stop_words_removal: bool, perform_accents_removal: bool,
                 perform_stemming: bool):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table = "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"
        # altere a linha abaixo para remoção de acentos (Atividade 11)
        self.accents_translation_table = str.maketrans(in_table, out_table)
        self.set_punctuation = set(string.punctuation)

        # flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self, html_doc: str) -> str:
        soup = BeautifulSoup(html_doc, features="lxml")
        return soup.get_text()

    @staticmethod
    def read_stop_words(str_file) -> set:
        set_stop_words = set()
        with open(str_file, encoding='utf-8') as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words

    def is_stop_word(self, term: str):
        return contains(self.set_stop_words, term.lower())

    def word_stem(self, term: str):
        return self.stemmer.stem(term)

    def remove_accents(self, term: str) -> str:
        return term.translate(self.accents_translation_table)

    def preprocess_word(self, term: str) -> str or None:
        if term in self.set_punctuation:
            return None

        if self.perform_stop_words_removal and self.is_stop_word(term):
            return None

        term = term.lower()

        if self.perform_accents_removal:
            term = self.preprocess_text(term)

        if self.perform_stemming:
            return self.word_stem(term)

        return term

    def preprocess_text(self, text: str) -> str or None:
        return self.remove_accents(text.lower())

class HTMLIndexer:
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                      language="portuguese",
                      perform_stop_words_removal=True,
                      perform_accents_removal=True,
                      perform_stemming=True)

    def __init__(self, index):
        self.index = index

    def text_word_count(self, plain_text: str):
        dic_word_count = {}
        tokens = word_tokenize(plain_text)

        for token in tokens:
            processed_token = self.cleaner.preprocess_word(token)
            if processed_token is not None:
                if not contains(dic_word_count.keys(), processed_token):
                    dic_word_count[processed_token] = 0

                dic_word_count[processed_token] += 1

        return dic_word_count

    def index_text(self, doc_id: int, text_html: str):
        clean_text = self.cleaner.html_to_plain_text(text_html)

        word_dict = self.text_word_count(clean_text)

        for key, value in word_dict.items():
            self.index.index(key, doc_id, value)

    def index_text_dir(self, path: str):
        for str_sub_dir in tqdm(os.listdir(path)):
        # for str_sub_dir in os.listdir(path):
            path_sub_dir = f"{path}/{str_sub_dir}"
            if os.path.isdir(path_sub_dir):
                self.index_text_dir(path_sub_dir)

            if path_sub_dir.endswith(".html"):
                with open(path_sub_dir, encoding="utf-8") as file:
                    content = file.read()
                    self.index_text(int(str_sub_dir.split(".")[0]), content)

        self.index.finish_indexing()
