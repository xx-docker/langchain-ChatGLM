# coding:utf-8 
import os 
import re 
# from pathlib import Path
DOC_DIR = "/home/cs110/cwp/"


def get_all_doc_filenames(dirname=DOC_DIR):
    doc_files = [] 
    for path, dir_list, file_list in  os.walk(dirname) :  
        for file_name in file_list:  
            __file = os.path.join(path, file_name)
            if re.match(".*?弃用文档.*?", __file):
                continue
            if re.match(".*\.(md|txt|pdf|doc|docx)", __file):
                doc_files.append(__file)
    return doc_files


if __name__ == '__main__':
    get_all_doc_filenames(dirname=DOC_DIR) 

