#_*_coding:utf-8_*_
import fileinput
from gensim import corpora
class comment_iterator(object):
    def __init__(self, input_filename,dict):
        self.filename = input_filename
        self.dict = dict
        print "打开分词后的文件：" + self.filename

    def __iter__(self):
        print "......生成格式化后评论的文件迭代器......"
        # print fileinput.input(self.filename)
        for line in fileinput.input(self.filename, mode='rb'):
            # assume there's one document per line, tokens separated by whitespace
            yield line