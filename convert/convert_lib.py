import collections
import numpy as np
from bert import tokenization
import json
import os

VOCAB_FILE = "/home/nnayak/mention_bottleneck/convert/cased_config_vocab/vocab.txt"
TOKENIZER = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)

class DatasetName(object):
  conll = 'conll12'
  preco = 'preco'

class Variation(object):
  classic = 'classic'
  sing = 'sing'
  gold = 'gold'
  goldconst = 'goldconst'
  predconst = 'predconst'

class DatasetSplit(object):
  train = 'train'
  dev = 'dev'
  test = 'test'
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


CLS = "[CLS]"
SPL = "[SPL]"
SEP = "[SEP]"


class Dataset(object):
  def __init__(self, dataset_name):
    self.name = dataset_name
    self.documents = collections.defaultdict(list)

  def dump_to_jsonl(self, file_name):
    assert file_name.endswith(".jsonl")
    
    assert ProcessingStage.TOKENIZED in self.documents
    if ProcessingStage.BPE_TOKENIZED not in self.documents:
      self.documents[ProcessingStage.BPE_TOKENIZED] = [
        bpe_tokenize_document(document, TOKENIZER)
        for document in self.documents[ProcessingStage.TOKENIZED]]

    for new_stage in [
      ProcessingStage.SEGMENTED_384, ProcessingStage.SEGMENTED_512]:

      self.documents[new_stage] = [
        segment_document(document, new_stage)
        for document in self.documents[ProcessingStage.BPE_TOKENIZED]]
      lines = [doc.dump_to_json() for doc in self.documents[new_stage]]

      seg_filename =  file_name.replace(".jsonl", "_" + new_stage + ".jsonl")
      with open(seg_filename, 'w') as f:
        f.write("\n".join(lines))


def flatten(nonflat):
  return sum(nonflat,[])


class ProcessingStage(object):
  UNINITIALIZED = "UNINITIALIZED"
  TOKENIZED = "TOKENIZED"
  BPE_TOKENIZED = "BPE_TOKENIZED"
  SEGMENTED_384 = "SEGMENTED_384"
  SEGMENTED_512 = "SEGMENTED_512"


class CorefDocument(object):
  def __init__(self, doc_id, part,
      initial_status=ProcessingStage.UNINITIALIZED):

    self.doc_id = doc_id
    self.doc_part = part
    self.status=initial_status

    self.clusters = []
    self.injected_mentions = []
    self.sentences = []
    self.speakers = []

    self.subtoken_map = []
    self.sentence_map = []

    self.other_info_json = None


  def dump_to_json(self):
    assert self.status in [ProcessingStage.SEGMENTED_512,
        ProcessingStage.SEGMENTED_384]
 
    return json.dumps({
          "doc_key": self.doc_id + "_" + str(int(self.doc_part)),
          "sentences": self.sentences,
          "sentence_map": self.sentence_map,
          "subtoken_map": self.subtoken_map,
          "speakers": self.speakers,
          "clusters": self.clusters,
          "inject_mentions": self.injected_mentions,
          "other_info": json.loads(self.other_info_json)
        })
    

def all_same(l):
  return len(set(l)) == 1

def bpe_tokenize_document(document, tokenizer):
  assert document.status == ProcessingStage.TOKENIZED

  bpe_document = CorefDocument(document.doc_id, document.doc_part,
    ProcessingStage.BPE_TOKENIZED)

  token_to_starting_subtoken = []
  token_to_ending_subtoken = []
  cum_doc_token_count = 0
  previous_token = 0

  for sentence_idx, sentence in enumerate(document.sentences):

    # All tokens in the sentence should have the same speaker, just checking
    multi_speakers = document.speakers[sentence_idx]
    assert all_same(multi_speakers)
    speaker, = tuple(set(multi_speakers))

    subword_list = [TOKENIZER.tokenize(token) for token in sentence]

    # Construct mapping to subtoken for use in cluster stuff later
    subtoken_offset = len(bpe_document.subtoken_map) + 1 # +1 for the CLS
    for i, (token, subwords) in enumerate(zip(sentence, subword_list)):
      token_to_starting_subtoken.append(subtoken_offset)
      subtoken_offset += len(subwords)
      token_to_ending_subtoken.append(subtoken_offset - 1) # inclusive

    subword_to_word = flatten(
      [
        [in_sentence_token_idx + cum_doc_token_count] * len(token_subwords)
        for in_sentence_token_idx, token_subwords in
        enumerate(subword_list)])

    flattened_subword = sum(subword_list, [])

    # Build various fields
    bpe_document.sentences.append(flattened_subword)
    bpe_document.sentence_map += [sentence_idx] * len(flattened_subword)  # fix this
    bpe_document.subtoken_map += subword_to_word
    bpe_document.speakers.append([speaker] * len(flattened_subword))

    cum_doc_token_count += len(sentence)
  assert all_same([len(token_to_ending_subtoken),
      len(token_to_starting_subtoken), len(flatten(document.sentences))])
           

  # Remap clusters
  for cluster in document.clusters:
    new_cluster = []
    for start, end in cluster:
      new_start = token_to_starting_subtoken[start]
      new_end = token_to_ending_subtoken[end]
      new_cluster.append([new_start, new_end])
    bpe_document.clusters.append(new_cluster)
  
  return bpe_document
  

STAGE_TO_LEN ={ProcessingStage.SEGMENTED_384: 384,
               ProcessingStage.SEGMENTED_512: 512}

def segment_document(bpe_document, new_stage):
  assert bpe_document.status == ProcessingStage.BPE_TOKENIZED
  max_segment_len = STAGE_TO_LEN[new_stage]
  seg_document = CorefDocument(
      bpe_document.doc_id, bpe_document.doc_part, new_stage)

  subtoken_offsets = []
  current_segment = []
  #doc_segments = []
  #doc_speakers = []
  current_speakers = []
  subtoken_offset = -1

  for i, (sentence, speakers) in enumerate(
    zip(bpe_document.sentences, bpe_document.speakers)):
    curr_sent_len = len(sentence)
    if (len(current_segment) + curr_sent_len + 2 >= max_segment_len
        or i == len(bpe_document.sentences) - 1):
      # A segment is complete, put it away and update the subtoken remap
      seg_document.sentences.append([CLS] + current_segment + [SEP])
      seg_document.speakers.append([SPL] + current_speakers + [SPL])
      subtoken_offset += 1 # for the CLS
      subtoken_offsets += [subtoken_offset] * len(current_segment)
      subtoken_offset += 1 # for the SEP
      
      current_segment = sentence
      current_speakers = speakers
    else:
      current_segment += sentence
      current_speakers += speakers


  #flat_sent = flatten(self.tokenized_sentences)
  #for i, k in enumerate(doc_segments):
  #  print("%%", i, len(k), k)
  #flat_seg = flatten(doc_segments)
  #seg_clusters = []
  #for cluster in self.clusters:
  #  new_cluster = []
  #  for start, end in cluster:
  #    print("^^", flat_sent[start-1:end])
  #    print(start, end)
  #    new_start, new_end = start + subtoken_offsets[start], end + subtoken_offsets[end]
  #    new_cluster.append([new_start, new_end])
  #    print("***", flat_seg[new_start:new_end+1])
  #    #print("---", flat_sent[start:end+1])
  #  print()
  #  seg_clusters.append(new_cluster)


  return seg_document
    

                           
    
def write_converted(dataset, prefix):
  dataset.dump_to_jsonl(prefix + ".jsonl")
