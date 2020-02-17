import conll_lib

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

    print("\n".join(str(i) for i in sent)) 
        
    
  

def main():
  # Read in one sentence at a tim
  # Add singleton labels

  filename = sys.argv[1]

  dataset = conll_lib.listify_conll_dataset(filename)

  conll_sing = conll_add_singletons(dataset)
  
  pass


if __name__ == "__main__":
  main()
