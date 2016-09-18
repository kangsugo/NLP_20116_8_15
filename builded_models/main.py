#_*_coding:utf-8_*_
'''
Created on 2016年8月3日

@author: sugo_yzk
'''
import jieba
import comment_iterator
import global_list
from Read_Comment import *
from MyCorpus import *
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
     #输入需要处理的文本
     demo = Read_Comment('test.txt')


     #对文本分词
     #实例化一个文本处理对象
     my_corpus = MyCorpus(demo)
     
     #得到字典
     #此处必须先执行一次，因为字典是遍历完才得到，不能边迭代产生每个单词的词袋模型边使用字典
     #memory_friendly不用全部存放在内存此函数不需要显式调用，在get_dict()中进行了隐式调用
     # my_corpus.get_Corpus()#得到one line one ducument,

     #使用迭代法产生dict时，必须显示调用该方法，因为字典是迭代出每一条评论产生的
     #但前提是one_line_one_comment必须存在
     my_corpus.get_Corpus()


     dic = my_corpus.get_dict()
     print "......得到词典......"
     print dic
     print type(dic)
     print dic.token2id
     for kkey in dic.token2id.keys():
          print kkey
          print  dic.token2id[kkey]
     # # print type(dic)
     # print "......得到各单词的id......"
     # print dic.token2id

     #必须得到词典之后才能调用
     comment = comment_iterator.comment_iterator("one_line_one_comment.txt",dic)

     for com in comment:#打印每一句话的词袋表示法
          print "句子是--->"
          print com.split()
          print "对应的词袋是。。。"
          print dic.doc2bow(com.split())#doc2bow只计算每个唯一单词的出现次数

     #语料库是
     corpus = [dic.doc2bow(text.split()) for text in comment]
     print corpus

     #训练这个语料库，使用tdifd模型，一旦初始化完成，可应用于任意语句向量
     tfidf = models.TfidfModel(corpus)
     #使用这个模型来变换向量
     print "------使用iftdf模型变换向量------"
     corpus_tfidf = tfidf[corpus]
     for doc in corpus_tfidf:
          print(doc)
		  
	 #打印的结果是[(1381, 0.8955385112309506), (1741, 0.1609712011177996), (6613, 0.41484822201974725)]
	 #即每个单词的权重，权重越大，代表这个word对文本意思的表达越重要
