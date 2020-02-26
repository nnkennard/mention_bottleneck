import collections
import conll_lib
import sys

NONSPAN = "NONSPAN"
VB_NONSPAN = "VB_NONSPAN"

def main():
  conll_file = sys.argv[1]

  list_dataset = conll_lib.listify_conll_dataset(conll_file)

  overall_span_counter = collections.Counter()
  coreferent_span_counter = collections.Counter()

  for document in list_dataset:
    for sentence in document[1:-1]:
      sequences = conll_lib.get_sequences(sentence)
      coref_clusters = conll_lib.build_coref_span_map(
          sequences[conll_lib.LabelSequences.COREF])
      all_coref_spans = sum(coref_clusters.values(), [])
      parse_spans = conll_lib.build_parse_span_map(
          sequences[conll_lib.LabelSequences.PARSE])

      nonspans = set(all_coref_spans) - set(parse_spans.keys())
      for start, end in nonspans:
        if start == end and sequences["POS"][start].startswith("VB"):
          coreferent_span_counter[VB_NONSPAN] += 1
        else:
          coreferent_span_counter[NONSPAN] += 1

      for span, label in parse_spans.items():
        condensed_label = label.replace("*", "")[1:-1]
        overall_span_counter[condensed_label] += 1
        if span in all_coref_spans:
          coreferent_span_counter[condensed_label] += 1 
      

  for d in [overall_span_counter, coreferent_span_counter]:
    for k, v in d.items():
      print(k+"\t"+str(v))
    print()



if __name__ == "__main__":
  main()
