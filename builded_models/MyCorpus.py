#_*_coding:utf-8_*_
import global_list
import jieba
from gensim import corpora
from six import iteritems
class MyCorpus(object):
     def __init__(self,ReadFile):
         self.demo = ReadFile

     #此处处理的是整个文档，即需要分析的所有数据
     #首先得到语料库（print dictionary），得到 "unique tokens"
     def get_Corpus(self):
        # all_words = []
        # format_str_str = u""
        # print type(self.demo)
        for phrase  in self.demo:#迭代得到每一行的每一个评论
            format_words = []
            # print phrase
            # print (phrase.encode("utf-8"))
            str1 = phrase.split('@')[-1]
            print str1
            seg_list = jieba.lcut(str1)
            # print seg_list
            #print  str1.encode('gb18030')

            for st in seg_list:
                # print st.encode('utf-8')
                word = st
                if word.find(global_list.b1)==-1 and\
                     word.find(global_list.b2)==-1 and\
                     word.find(global_list.b3)==-1 and\
                     word.find(global_list.b4)==-1 and\
                     word.find(global_list.b5)==-1 and\
                     word.find(global_list.b6)==-1 and\
                     word.find(global_list.b7)==-1 and\
                     word.find(global_list.b8)==-1 and\
                     word.find(global_list.b9)==-1 and\
                     word.find(global_list.b10)==-1 and\
                     word.find(global_list.b11)==-1 and\
                     word.find(global_list.b12)==-1 and\
                     word.find(global_list.b13)==-1 and\
                     word.find(global_list.b14)==-1 and\
                     word.find(global_list.b15)==-1 and\
                     word.find(global_list.b16)==-1 and\
                     word.find(global_list.b17)==-1 and \
                     word.find(global_list.b18) == -1 and \
                     word.find(global_list.b19) == -1 and \
                     word.find(global_list.b20) == -1 and \
                     word.find(global_list.b21) == -1 and \
                     word.find(global_list.b22)==-1 :
                     format_words.append(word.encode('utf-8'))

        # print format_words
        # print type(format_words)
        # print len(format_words)
        # for dd in format_words:
        #     print dd

            # format_word 是一句评论，即one line per document
            # all_word是列表的列表，每个列表元素是一个ducument

            with open("one_line_one_comment.txt","a") as ff:
                for one_word in format_words:
                    ff.write(one_word+'  ')
                ff.write('\n')


            # all_words.append(format_words)
        # return all_words

     # 得到分词结果。将每句话放入，得到字典。得到唯一的分词。
     def get_dict(self):
        stoplist = set('的'.split())
        print "停用词---->"
        print stoplist
        dictionary = corpora.Dictionary(line.split() for line in open('one_line_one_comment.txt'))
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
        once_ids=[]
        # once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids)
        dictionary.compactify()  # remove gaps in id sequence after words that were removed
        return dictionary

     #根据得到的语料库，将每一句话表示成为词袋模型：【（单词，出现次数），。。。。。。】
     #new_vec = dictionary.doc2bow(new_doc.split())


