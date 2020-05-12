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
    
    assert ProcessingStage.TOKENIZED in self.documents

    if ProcessingStage.BPE_TOKENIZED not in self.documents:
      self.documents[ProcessingStage.BPE_TOKENIZED] = [
        bpe_tokenize_document(document, TOKENIZER)
        for document in self.documents[ProcessingStage.TOKENIZED]]

    
    for new_stage in [
      ProcessingStage.SEGMENTED_384, ProcessingStage.SEGMENTED_512]:
      if new_stage not in self.documents:
        self.documents[new_stage] = [
          segment_document(document, new_stage)
          for document in self.documents[ProcessingStage.BPE_TOKENIZED]]

      lines = [doc.dump_to_json() for doc in self.documents[new_stage]]
    
      assert file_name.endswith(".jsonl")
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
  def __init__(self, doc_id, part, other_info="{}",
      init_status=ProcessingStage.UNINITIALIZED):

    self.doc_id = doc_id
    self.doc_part = part
    self.status=init_status

    self.clusters = []
    self.injected_mentions = []
    self.sentences = []
    self.speakers = []

    self.subtoken_map = []
    self.sentence_map = []

    self.other_info_json = other_info


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
          "other_info": json.loads(self.other_info_json),
          "format": self.status,
        })
    

def all_same(l):
  return len(set(l)) == 1

def same_len(l):
  return all_same(len(i) for i in l)

def bpe_tokenize_document(document, tokenizer):
  assert document.status == ProcessingStage.TOKENIZED

  bpe_document = CorefDocument(document.doc_id, document.doc_part,
      document.other_info_json, ProcessingStage.BPE_TOKENIZED)

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
    subtoken_offset = len(bpe_document.subtoken_map) # subtokens included so far
    for i, (token, subwords) in enumerate(zip(sentence, subword_list)):
      token_to_starting_subtoken.append(subtoken_offset)
      subtoken_offset += len(subwords)
      token_to_ending_subtoken.append(subtoken_offset - 1) # inclusive

    # For each subword, which original token did it come from (index from flat list)
    subtoken_map = flatten(
      [
        [in_sentence_token_idx + cum_doc_token_count] * len(token_subwords)
        for in_sentence_token_idx, token_subwords in
        enumerate(subword_list)])

    flattened_subword = sum(subword_list, [])

    # Build various fields
    bpe_document.sentences.append(flattened_subword)
    bpe_document.subtoken_map += subtoken_map
    bpe_document.speakers.append([speaker] * len(flattened_subword))

    # For each subtoken, which sentence did it come from (by idx)
    bpe_document.sentence_map += [sentence_idx] * len(flattened_subword) 

    cum_doc_token_count += len(sentence)

  assert same_len([token_to_ending_subtoken, token_to_starting_subtoken,
                   flatten(document.sentences)])
           

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
      bpe_document.doc_id, bpe_document.doc_part, bpe_document.other_info_json,
      new_stage)

  # For each segment, a list of sentence indices which are part of that segment
  segment_maps = []
  current_segment = []
  current_segment_len = 0
  
  # Building segment maps
  for i, sentence in enumerate(bpe_document.sentences):
    # 2 is added for CLS and SEP
    if len(sentence) + current_segment_len + 2 <= max_segment_len:
      current_segment.append(i)
      current_segment_len += len(sentence)
    else:
      segment_maps.append(current_segment)
      current_segment = [i]
      current_segment_len = len(sentence)
  if current_segment:
    segment_maps.append(current_segment)


  # For each subtoken, how many indices it is bumped by due to CLS and SEP tokens
  subtoken_offsets = []
  subtoken_offset = 0

  for sentence_indices in segment_maps:
    segment = [CLS] + flatten(bpe_document.sentences[i] for i in sentence_indices) + [SEP]
    speakers = [SPL] + flatten(bpe_document.speakers[i] for i in sentence_indices) + [SPL]
    seg_document.sentences.append(segment)
    seg_document.speakers.append(speakers)

    subtoken_offset += 1 # for the CLS token
    subtoken_offsets += [subtoken_offset] * (len(segment) - 2)
    subtoken_offset += 1 # for the SEP token
   
  # Remap clusters
  for cluster in bpe_document.clusters:
    new_cluster = []
    for start, end in cluster:
      new_start = start + subtoken_offsets[start]
      new_end = end + subtoken_offsets[end]
      new_cluster.append([new_start, new_end])
    seg_document.clusters.append(new_cluster)

  return seg_document
    
 
def write_converted(dataset, prefix):
  dataset.dump_to_jsonl(prefix + ".jsonl")
