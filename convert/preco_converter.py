import os
import json
import random
import tqdm

import convert_lib
import preco_lib

PRECO = convert_lib.DatasetName.preco
PRECO_MULT = convert_lib.DatasetName.preco_mult
DUMMY_DOC_PART = '0'

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def condense_sentences(sentences):
  """I literally don't know what this is for..."""
  sentence_index_map = {}
  new_sentences = []
  modified_sentence_count = 0
  modified_sentence_offsets = {}
  token_count = 0
  clean_sentences = []
  for sentence in sentences:
    clean_sentence = []
    for token in sentence:
      if token in ["\x7f"]:
        clean_sentence.append(" ")
      else:
        clean_sentence.append(token)
    clean_sentences.append(clean_sentence)

  for i, sentence in enumerate(clean_sentences):
    if len(sentence) == 1 and not sentence[0].strip():
      continue
    new_sentences.append(sentence)
    sentence_index_map[i] = modified_sentence_count
    modified_sentence_offsets[modified_sentence_count] = token_count
    token_count += len(sentence)
    modified_sentence_count += 1
  return new_sentences, sentence_index_map, modified_sentence_offsets
      
def make_empty_speakers(sentences):
  return [["" for token in sent] for sent in sentences]

def create_dataset(filename):
  dataset = convert_lib.Dataset(PRECO)
  lines = get_lines_from_file(filename)

  for line in tqdm.tqdm(lines):
    orig_document = json.loads(line)
    new_document = convert_lib.Document(
        convert_lib.make_doc_id(PRECO, orig_document["id"]), DUMMY_DOC_PART)
    sentence_offsets = []
    token_count = 0
  
    new_sentences, sentence_index_map, sentence_offsets = condense_sentences(
        orig_document["sentences"])
  
    new_document.sentences = new_sentences
    new_document.speakers = make_empty_speakers(new_document.sentences)
    new_document.clusters = []
    for cluster in orig_document["mention_clusters"]:
        new_cluster = []
        for sentence, begin, end in cluster:
          modified_sentence = sentence_index_map[sentence]
          new_cluster.append([sentence_offsets[modified_sentence] + begin,
          sentence_offsets[modified_sentence] + end - 1])
        new_document.clusters.append(new_cluster)
    dataset.documents.append(new_document)

  return dataset

def convert_subdataset(data_home, dataset_name):
  input_directory = os.path.join(data_home, "original", dataset_name)
  output_directory = os.path.join(data_home, "processed", dataset_name)
  convert_lib.create_dir(output_directory)
  preco_datasets = {}
  for split in [convert_lib.DatasetSplit.train, convert_lib.DatasetSplit.dev,
    convert_lib.DatasetSplit.test]:
    input_filename = os.path.join(input_directory, split + "." +
        convert_lib.FormatName.jsonl)
    converted_dataset = create_dataset(input_filename)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
    preco_datasets[split] = converted_dataset
 
def convert(data_home):
  preco_lib.convert(data_home)
  convert_subdataset(data_home, convert_lib.DatasetName.preco)
  convert_subdataset(data_home, convert_lib.DatasetName.preco_mult)
