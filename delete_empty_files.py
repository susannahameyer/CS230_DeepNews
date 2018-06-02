import os
import numpy as np

def delete_empty_files(folder_name):
	article_file_list = os.listdir(folder_name)
	article_file_list = sorted([ x for x in article_file_list if "label" not in x ], key=lambda x: int(x.split('.')[0]))
	for i in range(len(article_file_list)):
		filename = folder_name + "/" + article_file_list[i]
		label_filename = filename.split('.')[0] + "_label.txt"
		article = np.loadtxt(filename, dtype=np.str)
		# print article.type
		if article.size == 0:
			print filename
			print label_filename
			os.remove(filename)
			os.remove(label_filename)


delete_empty_files("train")