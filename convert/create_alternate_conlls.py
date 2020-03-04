import conll_lib
import convert_lib

import os
import re
import sys

NP_REGEX = r"\(NP\**\)"
VB_MARKABLES = ["VB", "VBG", "VBN", "VBP", "VBZ"]
TOK_MARKABLES = ["PRP$", "NNP", "NN", "CD", "PRP", "NNS", "NNPS", "DT",
                 "JJ", "CC", "MD"]


def get_pos_markables(pos_sequence, markable_list):
  pos_markables = []
  for i, pos in enumerate(pos_sequence):
    if pos in markable_list:
      pos_markables.append((i, i))
  return pos_markables


def conll_add_singletons(dataset, fn):
  """
    Args:
      dataset: listified dataset
  """
  new_dataset = []
  for document in dataset:
    new_dataset.append(doc_apply_labels(document, get_doc_labels(document, fn))) 
  return new_dataset


def create_additional_labels(new_mentions, cluster_offset, doc_len):
  labels = ["" for _ in range(doc_len)]
  for i , (start, end) in enumerate(sorted(new_mentions)):
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

def get_const_sent(unused_coref_map, parse_map, unused_pos):
  return set(parse_map.keys())

def get_gold_sent(coref_map, unused_parse_map, unused_pos):
  return set(sum(coref_map.values(), []))

def get_constgold_sent(coref_map, parse_map, unused_pos):
  return set(sum(coref_map.values(), [])).intersection(set(parse_map.keys()))

def get_consttok_sent(coref_map, parse_map, pos):
  janky_token_indices = set(sum((list(i) for i in parse_map.keys()), []))
  token_spans = [(i, i) for i in janky_token_indices]
  return set(parse_map.keys()).union(set(token_spans))

def get_constvb_sent(coref_map, parse_map, pos):
  return set(parse_map.keys()).union(set(get_pos_markables(pos, VB_MARKABLES)))


def get_npvbsing_sent(coref_map, parse_map, pos):
  coreferent_spans = set(sum(coref_map.values(), []))
  parse_markables = [span
                        for span, label in parse_map.items()
                        if re.match(NP_REGEX, label)]
  return set(
             parse_markables).union(
             coreferent_spans).union(set(get_pos_markables(pos, VB_MARKABLES)))



def get_nptoksing_sent(coref_map, parse_map, pos):
  coreferent_spans = set(sum(coref_map.values(), []))
  parse_markables = [span
                        for span, label in parse_map.items()
                        if re.match(NP_REGEX, label)]
  return set(
             parse_markables).union(
             coreferent_spans).union(set(get_pos_markables(pos, TOK_MARKABLES)))


def get_npsing_sent(coref_map, parse_map, unused_pos):
  coreferent_spans = set(sum(coref_map.values(), []))
  parse_markables = [span
                        for span, label in parse_map.items()
                        if re.match(NP_REGEX, label)]
  return set(parse_markables).union(coreferent_spans)

def get_doc_labels(document, sentence_fn):
 
  new_mentions = set()
  sentence_offset = 0
  mention_start_index = 0
  
  for sent in document[1:-1]:
    sequences = conll_lib.get_sequences(sent)
    coref_map = conll_lib.build_coref_span_map(
                    sequences["COREF"], sentence_offset)
    parse_map = conll_lib.build_parse_span_map(
                    sequences["PARSE"], sentence_offset)
  
    additional_mentions = sentence_fn(coref_map, parse_map, sequences["POS"])
    new_mentions.update(additional_mentions)
    sentence_offset += len(sequences["WORD"])
    maybe_mention_start_index = max(list(
      int(i) for i in coref_map.keys()) + [0]) + 1
    mention_start_index = max(mention_start_index, maybe_mention_start_index)

  doc_len = sentence_offset

  return create_additional_labels(new_mentions,
      mention_start_index, doc_len)


FN_MAP = {
  convert_lib.DatasetName.conll_npsing: get_npsing_sent,
  convert_lib.DatasetName.conll_nptoksing: get_nptoksing_sent,
  convert_lib.DatasetName.conll_npvbsing: get_npvbsing_sent,
  convert_lib.DatasetName.conll_const: get_const_sent,
  convert_lib.DatasetName.conll_constvb: get_constvb_sent,
  convert_lib.DatasetName.conll_consttok: get_consttok_sent,
  convert_lib.DatasetName.conll_constgold: get_constgold_sent,
  convert_lib.DatasetName.conll_gold: get_gold_sent
}


def doc_apply_labels(document, labels):
  coref_idx = conll_lib.CONLL_FIELD_MAP[conll_lib.LabelSequences.COREF]
  for sent in document[1:-1]:
    for i, word in enumerate(sent):
      curr_coref_label = word[coref_idx]
      label = labels.pop(0)
      if not label:
        continue
      else:
        if curr_coref_label == "-":
          sent[i][coref_idx] = label
        else:
          sent[i][coref_idx] += "|" + label
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

  for new_dataset_name in [convert_lib.DatasetName.conll_npsing,
                           convert_lib.DatasetName.conll_const,
                           convert_lib.DatasetName.conll_consttok,
                           convert_lib.DatasetName.conll_constvb,
                           convert_lib.DatasetName.conll_npvbsing,
                           convert_lib.DatasetName.conll_nptoksing,
                           convert_lib.DatasetName.conll_constgold,
                           convert_lib.DatasetName.conll_gold]:
    input_directory = os.path.join(
        data_home, "original", convert_lib.DatasetName.conll)
    output_directory = os.path.join(
        data_home, "original", new_dataset_name)
    convert_lib.create_dir(output_directory)

    for split in convert_lib.DatasetSplit.ALL:
      input_filename = os.path.join(input_directory, "conll12_" + split + ".txt")
      output_filename = os.path.join(output_directory, "conll12_" + split + ".txt")
      dataset = conll_lib.listify_conll_dataset(input_filename)
      fn = FN_MAP[new_dataset_name]
      new_dataset = conll_add_singletons(dataset, fn)
      write_conll_to_file(new_dataset, output_filename)

if __name__ == "__main__":
  main()
