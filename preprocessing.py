#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import jieba
import itertools
from collections import Counter
import h5py
import numpy as np

filepath = '/Users/zhangjun/Desktop/Torch/dataset/LCSTS/DATA/'
topN = 5000
sourceSeqLen = 80
targetSeqLen = 20
f = h5py.File("seq2seq.hdf5","w")

def extract(filename):
	filename = filepath + filename
	with open(filename) as f:
		text = f.read()
		summary = []
		short_text = []
		_summary = re.findall("<summary>([^<]+)</summary>",text)
		_short_text = re.findall("<short_text>([^<]+)</short_text>",text)
		summary = [s.strip() for s in _summary]
		short_text = [s.strip() for s in _short_text]
		return short_text,summary

def tokenize(text,_type):
	if _type == "s":
		seqLen = sourceSeqLen
	else:
		seqLen = targetSeqLen
	seg_list = jieba.cut(text,cut_all=False)
	s =  " ".join(seg_list)
	s_length = len(s.split(" "))
	if s_length < (seqLen - 2):
		s += (seqLen - 2 - s_length) * " <blank>"
		s = s.split(" ")
	else:
		s = s.split(" ")
		s = s[:seqLen-2]
	t = ["<s>"] + s + ["</s>"]
	return t

def buildCorpus(filename="PART_III.txt"):
	jieba.enable_parallel(4)
	source,target = extract(filename)
	source = [tokenize(s,"s") for s in source]
	target = [tokenize(t,"t") for t in target]
	return source,target

def buildVocab(sents):
	word_counts = Counter(itertools.chain(*sents))
	vocabulary_inv = [x[0] for x in word_counts.most_common()[:topN-1]]
	vocabulary = {}
	vocabulary = {x: i+2 for i, x in enumerate(vocabulary_inv)} 
	vocabulary["<unk>"] = 1
	vocabulary_txt = sorted(vocabulary.iteritems(), key=lambda d:d[1], reverse = False)
	vf = file("vocab.dict","w")
	for word,_ in vocabulary_txt:
		vf.write(word.encode("utf-8"))
		vf.write("\n")
	vf.close()
	return vocabulary # dict

def buildSents(_sents,vocabulary):
	sents = []
	for _sent in _sents:
		sents.append([vocabulary[w] if w in vocabulary else vocabulary['<unk>'] for w in _sent])
	return sents

if __name__ == "__main__":
	source,target = buildCorpus()
	sents = source + target
	vocabulary = buildVocab(sents)
	source = buildSents(source,vocabulary)
	target = buildSents(target,vocabulary)
	f["source"] = np.array(source)
	f["target"] = np.array(target)
	f.close()
	# print(vocabulary)
	# print(sents)
	# print("test corpus done!")
	# print("building train corpus....")
	# train_source,train_target = buildCorpus("PART_I.txt")
	# print("train corpus done!")
	# sents  = test_source + test_target + train_source + train_target
	# print("building vocab.....")
	# vocabulary = buildVocab(sents)
	# print("vocab done!")
	# print("replace words by id.........")
	# print("...............test source ...................")
	# test_source = buildSents(test_source,vocabulary)
	# print("..................test target................")
	# test_target = buildSents(test_target,vocabulary)
	# print(".................train target.................")
	# train_target = buildSents(train_target,vocabulary)
	# print(".................train source.................")
	# train_source = buildSents(train_source,vocabulary)
	# print("writing to files")
	# writeToFile(test_source,"test_source.txt")
	# writeToFile(test_target,"test_target.txt")
	# writeToFile(train_source,"train_source.txt")
	# writeToFile(train_target,"train_target.txt")
	# print("end......")




	






	
	