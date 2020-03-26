import json
import os


class DatasetName(object):
  conll = 'conll'
  preco = 'preco'

class Variation(object):
  classic = 'classic'
  sing = 'sing'
  gold = 'gold'
  goldconst = 'goldconst'
  predconst = 'predconst'

class DatasetSplit(object):
  train = 'train'
  test = 'test'
  dev = 'dev'
  ALL = [train, dev, test]


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


class Document(object):
  def __init__(self, doc_id, doc_part):
    self.doc_id = doc_id
    self.doc_part = doc_part
    self.doc_key = "UNK"
    self.sentences = []
    self.speakers = []
    self.clusters = []
    self.injected_mentions = []
    self.parse_spans = []
    self.pos = []
    self.singletons = []

  def dump_to_jsonl(self):

    nonsingleton_clusters = [
      cluster for cluster in self.clusters if len(cluster) > 1]

    return [json.dumps({
          "doc_key": self.doc_id + "_" + str(int(self.doc_part)),
          "document_id": self.doc_id + "_" + self.doc_part,
          "sentences": self.sentences,
          "speakers": self.speakers,
          "clusters": nonsingleton_clusters,
          "inject_mentions": self.injected_mentions,
          "parse_spans": self.parse_spans,
          "pos": self.pos
        })]

def write_converted(dataset, prefix):
  dataset.dump_to_jsonl(prefix + ".jsonl")
