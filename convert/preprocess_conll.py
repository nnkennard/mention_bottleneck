import conll_lib
import convert_lib

import os
import re
import sys

MARKABLES_REGEX = r"\(NP\**\)"

POS_MARKABLES = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "PRP", "PRP$"]

def get_pos_markables(pos_sequence):
  pos_markables = []
  for i, pos in enumerate(pos_sequence):
    if pos in POS_MARKABLES:
      pos_markables.append((i, i))
  return pos_markables



def conll_add_singletons(dataset):
  """
    Args:
      dataset: listified dataset
  """
  new_dataset = []
  for document in dataset:
    new_dataset.append(doc_add_singletons(document)) 
  return new_dataset


def create_singleton_labels(singletons, cluster_offset, doc_len):
  labels = ["" for _ in range(doc_len)]
  for i , (start, end) in enumerate(sorted(singletons)):
    if labels[start]:
      labels[start] += "|"
    if start == end:
      labels[start] += "(s" + str(i + cluster_offset) + ")"
    else:
      labels[start] += "(s"+ str(i + cluster_offset) 
      if labels[end]:
        labels[end] += "|"
      labels[end] +=  "s" + str(i + cluster_offset) + ")" 
  return labels


def doc_add_singletons(document):
  """
    Args:
      document: listified document
  """
  sentence_offset = 0

  doc_singletons = set()
  coreferent_entity_nums = set()
  
  for sent_i, sent in enumerate(document[1:-1]):
    sequences = conll_lib.get_sequences(sent)
    coref_map = conll_lib.build_coref_span_map(
                    sequences["COREF"], sentence_offset)
    parse_map = conll_lib.build_parse_span_map(
                    sequences["PARSE"], sentence_offset)

    parse_markables = [span
                          for span, label in parse_map.items()
                          if re.match(MARKABLES_REGEX, label)]
    pos_markables = [(sentence_offset + i, sentence_offset + i)
                      for i, pos in enumerate(sequences["POS"])
                      if pos in POS_MARKABLES]
    
    coreferent_spans = set(sum(coref_map.values(), []))
    coreferent_entity_nums.update(int(i) for i in coref_map.keys())
    singletons = set(parse_markables).union(set(pos_markables)) - coreferent_spans
    doc_singletons.update(list(singletons))

    sentence_offset += len(sequences["WORD"])

  singleton_labels = create_singleton_labels(doc_singletons, max(list(coreferent_entity_nums)+ [0]), sentence_offset)

  coref_idx = conll_lib.CONLL_FIELD_MAP[conll_lib.LabelSequences.COREF]
  for sent in document[1:-1]:
    for i, word in enumerate(sent):
      curr_coref_label = word[coref_idx]
      singleton_label = singleton_labels.pop(0)
      if not singleton_label:
        continue
      else:
        if curr_coref_label == "-":
          sent[i][coref_idx] = singleton_label
        else:
          sent[i][coref_idx] += singleton_label

  return document

def write_conll_to_file(conll_list_dataset, output_filename):
  with open(output_filename, 'w') as f:
    for document in conll_list_dataset:
      for sentence in document:
        if sentence[0][0].startswith("#"):
          assert len(sentence) == 1
          f.write("\t".join(sentence[0]) + "\n")
        else:
          f.write("\n".join("\t".join(word) for word in sentence))
          f.write("\n\n")
          
def main():

  data_home = sys.argv[1]

  
  input_directory = os.path.join(
      data_home, "original", convert_lib.DatasetName.conll)
  output_directory = os.path.join(
      data_home, "original", convert_lib.DatasetName.conll_sing)
  convert_lib.create_dir(output_directory)

  for split in convert_lib.DatasetSplit.ALL:
    input_filename = os.path.join(input_directory, "conll12_" + split + ".txt")
    output_filename = os.path.join(output_directory, "conll12_" + split + ".txt")
    dataset = conll_lib.listify_conll_dataset(input_filename)
    conll_sing = conll_add_singletons(dataset)
    write_conll_to_file(conll_sing, output_filename)
if __name__ == "__main__":
  main()
