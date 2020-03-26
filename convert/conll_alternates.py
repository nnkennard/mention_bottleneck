import conll_lib
import convert_lib

import re

# Begin alternate conll functions

def get_gold_sent(coref_map, unused_parse_map, unused_pos):
  return set(sum(coref_map.values(), []))

def get_goldconst_sent(coref_map, parse_map, pos):
  janky_token_indices = set(sum((list(i) for i in parse_map.keys()), []))
  token_spans = [(i, i) for i in janky_token_indices]
  return set(parse_map.keys()).union(set(token_spans))

def get_sing_sent(coref_map, parse_map, pos):
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

FN_MAP = {
  convert_lib.DatasetName.sing: get_sing_sent,
  convert_lib.DatasetName.gold: get_gold_sent,
  convert_lib.DatasetName.goldconst: get_goldconst_sent,
  }

# End alternate conll functions


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
    cluster_id = "s" + str(i + cluster_offset)
    if labels[start]:
      labels[start] += "|"
    if start == end:
      labels[start] += "({0})".format(cluster_id)
    else:
      labels[start] += "({0}".format(cluster_id)
      if labels[end]:
        labels[end] += "|"
      labels[end] += "{0})".format(cluster_id)
  return labels

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
     
