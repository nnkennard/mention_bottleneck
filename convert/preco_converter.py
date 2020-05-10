import os
import json
import random
import tqdm

import convert_lib
import preco_lib

DUMMY_DOC_PART = '0'

def get_lines_from_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def condense_sentences(sentences):
  """Need to figure out what this actually does and why."""
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
  dataset = convert_lib.Dataset(convert_lib.DatasetName.preco)
  lines = get_lines_from_file(filename)

  for line in tqdm.tqdm(lines):
    orig_document = json.loads(line)
    new_document = convert_lib.CorefDocument(
        convert_lib.make_doc_id("preco", orig_document["id"]), DUMMY_DOC_PART,
            init_status=convert_lib.ProcessingStage.TOKENIZED)
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
    dataset.documents[convert_lib.ProcessingStage.TOKENIZED].append(new_document)

  return dataset

def convert_format(data_home):
  """Convert preco json format into spanbert json format."""
  input_directory = os.path.join(data_home, "original", "preco")
  output_directory = os.path.join(data_home, "processed", "preco/all_info")
  convert_lib.create_dir(output_directory)
  for split in [convert_lib.DatasetSplit.train, convert_lib.DatasetSplit.dev,
    convert_lib.DatasetSplit.test]:
    input_filename = os.path.join(input_directory, split + ".jsonl")
    converted_dataset = create_dataset(input_filename)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
 

def get_examples(filename):
  with open(filename, 'r') as f:
    return [json.loads(line) for line in f.readlines()]

def get_sing_injected(example):
  return sum(example["clusters"], [])

def get_gold_injected(example):
  return sum(
    [cluster for cluster in example["clusters"] if len(cluster) > 1], [])

def get_goldsing_injected(example):
  return []

def get_classic_injected(example):
  return []

FN_MAP = {
  "sing": get_sing_injected,
  "gold": get_gold_injected,
  "goldsing": get_goldsing_injected,
  "classic": get_classic_injected
}

def keep_singletons(inject_type):
  return inject_type == "goldsing"

def create_injected_file(superset_filename, inject_type):
  examples = get_examples(superset_filename)
  for example in examples:
    example["injected_mentions"] = FN_MAP[inject_type](example)
    if not keep_singletons(inject_type):
      clusters = [cluster for cluster in example["clusters"] if len(cluster) >1]
      example["clusters"] = clusters
      
  out_file = superset_filename.replace("all_info", inject_type)
  convert_lib.create_dir("/".join(out_file.split("/")[:-1]))
  with open(out_file, 'w') as f:
    f.write("\n".join(json.dumps(example) for example in examples))


def convert(data_home):
  # Just makes train-test splits
  preco_lib.preprocess(data_home)

  convert_format(data_home)
  superset_dir = os.path.join(data_home, "processed", "preco/all_info")
  for subset in ["train", "dev", "test"]:
    all_info_filename = superset_dir + "/" + subset + ".jsonl"
    for new_type in FN_MAP.keys():
      convert_lib.create_dir(superset_dir + "/" + new_type)
      create_injected_file(all_info_filename, new_type)
