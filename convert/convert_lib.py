import tqdm
import json

import collections
import csv
import json
import os
import numpy as np


class DatasetName(object):
  conll = 'conll_mult' 
  conll_sing = 'conll_sing'
  conll_npsing = 'conll_npsing'
  conll_nptoksing = 'conll_nptoksing'
  conll_npvbsing = 'conll_npvbsing'
  conll_const = 'conll_const'
  conll_consttok = 'conll_consttok'
  conll_constvb = 'conll_constvb'
  conll_gold = 'conll_gold'
  conll_constgold = 'conll_constgold'


  preco = 'preco_sing'
  preco_mult = 'preco_mult'

  #ALL_DATASETS = [
  #    conll, conll_sing, conll_npsing, conll_const, conll_gold,
  #    preco, preco_mult]


class DatasetSplit(object):
  train = 'train'
  test = 'test'
  dev = 'dev'
  ALL = [train, dev, test]


class FormatName(object):
  jsonl = 'jsonl'
  file_per_doc = 'file_per_doc'
  ALL_FORMATS = [jsonl, file_per_doc]

def create_dir(path):
  try:
      os.makedirs(path)
  except OSError:
      print ("Creation of the directory %s failed" % path)
  else:
      print ("Successfully created the directory %s " % path)

def make_doc_id(dataset, doc_name):
  if dataset.startswith('preco'):
    dataset = "nw_" + dataset # Placeholder domain for preco 
  if type(doc_name) == list:
    doc_name = "_".join(doc_name)
  return "_".join([dataset, doc_name])


class Dataset(object):
  def __init__(self, dataset_name):
    self.name = dataset_name
    self.documents = []

  def dump_to_jsonl(self, file_name):
    lines = []
    for doc in self.documents:
      lines += doc.dump_to_jsonl()
    with open(file_name, 'w') as f:
      f.write("\n".join(lines))

  def dump_to_fpd(self, directory):
    create_dir(directory)
    for doc in tqdm.tqdm(self.documents):
      with open(
        directory + "/" + doc.doc_id.replace("/", "-") + "_" + doc.doc_part + ".txt", 'w') as f:
        f.write("\n".join(doc.dump_to_fpd()))


def flatten(l):
  return sum(l, [])


class Document(object):
  def __init__(self, doc_id, doc_part):
    self.doc_id = doc_id
    self.doc_part = doc_part
    self.doc_key = "UNK"
    self.sentences = []
    self.speakers = []
    self.clusters = []
    self.additional_mentions = []
    self.parse_spans = []
    self.pos = []
    self.singletons = []

    self.label_sequences = {}

  def dump_to_fpd(self):
    return [" ".join(sentence) for sentence in self.sentences]

  def dump_to_jsonl(self):

    nonsingleton_clusters = [
      cluster for cluster in self.clusters if len(cluster) > 1]

    return [json.dumps({
          "doc_key": self.doc_id + "_" + str(int(self.doc_part)),
          "document_id": self.doc_id + "_" + self.doc_part,
          "sentences": self.sentences,
          "speakers": self.speakers,
          "clusters": nonsingleton_clusters,
          "additional_mentions": self.additional_mentions,
        })]

def write_converted(dataset, prefix):
    dataset.dump_to_fpd(prefix + "-fpd/")
    dataset.dump_to_jsonl(prefix + ".jsonl")
