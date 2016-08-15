#_*_coding:utf-8_*_
import fileinput
import string

class Read_Comment(object):
     def __init__(self,input_filename):
         self.filename = input_filename
         print "打开文件："+self.filename


     def __iter__(self):
         print "......生成文件迭代器......"
         # print fileinput.input(self.filename)
         for line in fileinput.input(self.filename,mode='rb'):
             # assume there's one document per line, tokens separated by whitespace
             yield line.decode("utf-8")
