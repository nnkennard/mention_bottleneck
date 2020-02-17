import tqdm
import json

import collections
import csv
import json
import os
import numpy as np


class DatasetName(object):
  conll = 'conll' 
  preco = 'preco'
  preco_mult = 'preco_mult'
  ALL_DATASETS = [conll, preco, preco_mult]


class DatasetSplit(object):
  train = 'train'
  test = 'test'
  dev = 'dev'
  ALL = [train, dev, test]


class FormatName(object):
  jsonl = 'jsonl'
  file_per_doc = 'file_per_doc'
  ALL_FORMATS = [jsonl, file_per_doc]

def get_filename(data_home, dataset_name, dataset_split, format_name):
  return os.path.join(data_home, 'processed', dataset_name,
      dataset_split + "." + format_name)

def create_dir(path):
  try:
      os.makedirs(path)
  except OSError:
      print ("Creation of the directory %s failed" % path)
  else:
      print ("Successfully created the directory %s " % path)

def make_doc_id(dataset, doc_name):
  if type(doc_name) == list:
    doc_name = "_".join(doc_name)
  return "_".join([dataset, doc_name])
  assert False


class Dataset(object):
  def __init__(self, dataset_name):
    self.name = dataset_name
    self.documents = []

  def _dump_lines(self, function, file_handle):
    lines = []
    for doc in self.documents:
      lines += doc.apply_dump_fn(function)
    file_handle.write("\n".join(lines))

  def dump_to_jsonl(self, file_handle):
    self._dump_lines("jsonl", file_handle)

  def dump_to_fpd(self, directory):
    create_dir(directory)
    for doc in tqdm.tqdm(self.documents):
      with open(
        directory + "/" + doc.doc_id + "_" + doc.doc_part + ".txt", 'w') as f:
        f.write("\n".join(doc.dump_to_fpd()))


def flatten(l):
  return sum(l, [])


class LabelSequences(object):
  WORD = "WORD"
  POS = "POS"
  NER = "NER"
  PARSE = "PARSE"
  COREF = "COREF"
  SPEAKER = "SPEAKER"


class Document(object):
  def __init__(self, doc_id, doc_part):
    self.doc_id = doc_id
    self.doc_part = doc_part
    self.doc_key = "UNK"
    self.sentences = []
    self.speakers = []
    self.clusters = []
    self.parse_spans = []
    self.pos = []
    self.singletons = []

    self.label_sequences = {}

    self.FN_MAP = {
      "jsonl": self.dump_to_jsonl,
      "file_per_doc": self.dump_to_fpd}

  def dump_to_fpd(self):
    return [" ".join(sentence) for sentence in self.sentences]


  def apply_dump_fn(self, function):
    return self.FN_MAP[function]()

  def _get_conll_coref_labels(self):
    coref_labels = collections.defaultdict(list)
    for cluster, tok_idxs in enumerate(self.clusters):
      for tok_start, tok_end in tok_idxs:
        if tok_start == tok_end:
          coref_labels[tok_start].append("({})".format(cluster))
        else:
          coref_labels[tok_start].append("({}".format(cluster))
          coref_labels[tok_end].append("{})".format(cluster))

    return coref_labels

  def remove_singletons(self):
    new_clusters = []
    for cluster in self.clusters:
      if len(cluster) > 1:
        new_clusters.append(cluster)
    self.clusters = new_clusters

  def dump_to_jsonl(self):

    return [json.dumps({
          "doc_key": self.doc_id + "_" + self.doc_part,
          "document_id": self.doc_id + "_" + self.doc_part,
          "sentences": self.sentences,
          "speakers": self.speakers,
          "clusters": self.clusters,
          "parse_spans": self.parse_spans,
          "pos": self.pos
        })]


def write_converted(dataset, prefix):
    dataset.dump_to_fpd(prefix + "-fpd/")
    with open(prefix + ".jsonl", 'w') as f:
      dataset.dump_to_jsonl(f)
