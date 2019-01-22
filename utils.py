import numpy
from collections import deque
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

class InputData:
    """Process input data into word2vec format for use.

    Attributes:
        word2id:
        id2word:
        word_frequency: 
    """
    def __init__(self, filename):
        self.filename = filename
        self.rawFile = self.load_data()
        self.fileArray = self.rawFile.split()
        self.vocab = set (self.fileArray)

    def load_data(self):
        with open(self.filename, "rb") as file:
            processsed_text = file.read().decode("utf-8").strip() #strip is to remove start and end chars in the file
            # print(type(processsed_text))
        return processsed_text
            
if __name__ == "__main__":
    dataset = InputData("medium_text.txt")
    print(dataset.vocab)


