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
    self.documents = []

  def dump_to_jsonl(self, file_name):
    assert file_name.endswith(".jsonl")
    for max_segment_len in [384, 512]:
      seg_filename =  file_name.replace(
          ".jsonl", "_" + str(max_segment_len) + ".jsonl")
      lines = []
      for doc in self.documents:
        lines += doc.dump_to_jsonl(max_segment_len)
      with open(seg_filename, 'w') as f:
        f.write("\n".join(lines))


def flatten(nonflat):
  return sum(nonflat,[])


class ProcessingStage(object):
  UNINITIALIZED = 0
  TOKENIZED = 1
  BPE_TOKENIZED = 2
  SEGMENTED_384 = 3
  SEGMENTED_512 = 4


class CorefDocument(object):
  def __init__(self, doc_id, part,
      initial_status=ProcessingStage.UNINITIALIZED):

    self.doc_id = doc_id
    self.doc_part = part
    self.status=initial_status

    self.clusters = None
    self.injected_mentions = None
    self.sentences = None
    self.speakers = None

    self.other_info_json = None

    

def bpe_tokenize_document(document, tokenizer):
  assert document.status == ProcessingStage.TOKENIZED

  pass
  
  document.status = ProcessingStage.BPE_TOKENIZED

def segment_document(document, max_segment_len):
  assert document.status == BPE_TOKENIZED
  segmented_document = CorefDocument(
      document.doc_id, document.doc_part, ProcessingStage.UNINITIALIZED)

  return segmented_document
    

    
class Document(object):
  def __init__(self, doc_id, doc_part):
    #self.doc_id = doc_id
    #self.doc_part = doc_part
    self.sentences = []
    self.speakers = []
    self.clusters = {}
    self.injected_mentions = []
    self.parse_spans = []
    self.pos = []
    self.singletons = []
    
    self.bertified = False

  def bertify(self):
    # Bertify needs to bert tokenize the tokens and adjust the clusters.
    assert self.token_sentences is None

    # BERT-tokenized sentences
    self.token_sentences = self.sentences
    self.tokenized_sentences = []
    
    # Map from subword to sentence index
    self.sentence_map = []

    # Map from subtoken to original word index
    self.subtoken_map = []

    token_to_starting_subtoken = []
    token_to_ending_subtoken = []
    cum_doc_token_count = 0
    previous_token = 0

    for sentence_idx, sentence in enumerate(self.token_sentences):

      # All tokens in the sentence should have the same speaker, just checking
      multi_speakers = self.speakers[sentence_idx]
      assert len(set(multi_speakers)) == 1
      speaker, = tuple(set(multi_speakers))

      subword_list = [TOKENIZER.tokenize(token) for token in sentence]

      # Construct mapping to subtoken for use in cluster stuff later
      subtoken_offset = len(self.subtoken_map) + 1 # +1 for the CLS
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
      self.tokenized_sentences.append(flattened_subword)
      self.sentence_map += [sentence_idx] * len(flattened_subword)  # fix this
      self.subtoken_map += subword_to_word
      self.speakers.append([speaker] * len(flattened_subword))

      cum_doc_token_count += len(sentence)

    flat_subtokens = flatten(self.sentences)
    # Remap clusters
    self.token_clusters = self.clusters
    self.clusters = []
    for cluster in self.token_clusters:
      new_cluster = []
      for start, end in cluster:
        new_start = token_to_starting_subtoken[start]
        new_end = token_to_ending_subtoken[end]
        new_cluster.append([new_start, new_end])
        #print("((", flat_subtokens[start:end+1])
      
      self.clusters.append(new_cluster)

                       
    self.bertified = True
    self.segmentize(384)

  def segmentize(self, segment_len):
    subtoken_offsets = []
    current_segment = []
    doc_segments = []
    doc_speakers = []
    current_speakers = []
    subtoken_offset = -1
    assert self.bertified
    for i, (sentence, speakers) in enumerate(zip(self.tokenized_sentences, self.speakers)):
      curr_sent_len = len(sentence)
      if len(current_segment) + curr_sent_len + 2 >= segment_len or i == len(self.sentences) - 1:
        # A segment is complete, put it away and update the subtoken remap
        doc_segments.append([CLS] + current_segment + [SEP])
        doc_speakers.append([SPL] + current_speakers + [SPL])
        subtoken_offset += 1 # for the CLS
        subtoken_offsets += [subtoken_offset] * len(current_segment)
        subtoken_offset += 1 # for the SEP
        
        current_segment = sentence
        current_speakers = speakers
      else:
        current_segment += sentence
        current_speakers += speakers
  
    flat_sent = flatten(self.tokenized_sentences)
    for i, k in enumerate(doc_segments):
      print("%%", i, len(k), k)
    flat_seg = flatten(doc_segments)
    seg_clusters = []
    for cluster in self.clusters:
      new_cluster = []
      for start, end in cluster:
        print("^^", flat_sent[start-1:end])
        print(start, end)
        new_start, new_end = start + subtoken_offsets[start], end + subtoken_offsets[end]
        new_cluster.append([new_start, new_end])
        print("***", flat_seg[new_start:new_end+1])
        #print("---", flat_sent[start:end+1])
      print()
      seg_clusters.append(new_cluster)

    exit()
   

  def dump_to_jsonl(self, segment_len):

    if not self.bertified:
      self.bertify()

    all_subtokens = flatten(self.sentences)
    segments = []
    for i in range(0, len(all_subtokens), segment_len):
      segments.append(all_subtokens[i:i+segment_len])

    nonsingleton_clusters = [
      cluster for cluster in self.clusters if len(cluster) > 1]

    return [json.dumps({
          "doc_key": self.doc_id + "_" + str(int(self.doc_part)),
          "document_id": self.doc_id + "_" + self.doc_part,
          "sentences": segments,
          "sentence_map": self.sentence_map,
          "subtoken_map": self.subtoken_map,
          #"token_sentences": self.token_sentences,
          "speakers": self.speakers,
          "clusters": nonsingleton_clusters,
          #"token_clusters": self.token_clusters,
          "inject_mentions": self.injected_mentions,
          "parse_spans": self.parse_spans,
          "pos": self.pos
        })]

def write_converted(dataset, prefix):
  dataset.dump_to_jsonl(prefix + ".jsonl")
