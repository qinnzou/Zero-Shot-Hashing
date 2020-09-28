# -*- coding: utf-8 -*-


import numpy as np

f = open('words_300d.txt', 'r', encoding='utf-8')
all_words = f.readlines()
all_words = [word.strip() for word in all_words]

f1 = open('vectors_300d.txt', 'r', encoding='utf-8')
all_vectors = f1.readlines()
all_vectors = [vector.strip() for vector in all_vectors]


f2 = open('coco_concept60.txt', 'r', encoding='utf-8')

query_words = f2.readlines()
query_words = [word.strip() for word in query_words]
print(query_words)

count = 0
vectors = []
for query_word in query_words:
    if query_word in all_words:
        index = all_words.index(query_word)
        print(all_words[index])
        vector_c = all_vectors[index]
        # s = vector_c.split(' ')
        s = vector_c.split('\t')
        vector = [float(item) for item in s]
        print(vector)
        vectors.append(vector)
        count = count + 1

np.save('coco60_word2vec_300d.npy', vectors)

