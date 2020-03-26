import collections
import json
import sys


def read_antecedent_file(antecedent_file):
  antecedent_map = collections.defaultdict(list)
  with open(antecedent_file, 'r') as f:
    for line in f:
      if not line.startswith("ant"):
        continue
      fields = line.split()
      _, _, doc, start, end, initial_ant, _, revised_ant, _ = line.split()
      antecedent_map[doc].append(((start, end), initial_ant, revised_ant))
  return antecedent_map


def add_doc_key(doc_key, mentions):
  return [(doc_key, mention) for mention in sorted(list(mentions))]

def blerp(example, antecedents):

  mention_list, initial_ant, final_ant  = zip(*antecedents)
  gold_mentions = set(tuple(i) for i in sum(example["clusters"], []))
  additional_mentions = set(tuple(i) for i in example["additional_mentions"])
  gold_singletons = additional_mentions - gold_mentions
  final_clusters = create_clusters(mention_list, final_ant)
  print(final_clusters)
  print()

  unused_stuff = """
  true_pos_singletons = gold_singletons.intersection(final_singletons)
  true_neg_singletons = gold_mentions.intersection(final_coreferent)

  false_pos_singletons = gold_mentions.intersection(final_singletons)
  false_neg_singletons = gold_singletons.intersection(final_coreferent)

  return {
    "tp": add_doc_key(example["doc_key"], true_pos_singletons),
    "tn": add_doc_key(example["doc_key"], true_neg_singletons),
    "fp": add_doc_key(example["doc_key"], false_pos_singletons),
    "fn": add_doc_key(example["doc_key"], false_neg_singletons),
    }
"""


def main():
  consttok_json_file, antecedent_file = sys.argv[1:]
  antecedent_map = read_antecedent_file(antecedent_file)
  overall_mention_tracker = collections.defaultdict(list)
  with open(consttok_json_file, 'r') as f:
    for line in f:
      example = json.loads(line)
      example_errors = blerp(example, antecedent_map[example["doc_key"]])
      for k, v in example_errors.items():
        overall_mention_tracker[k] += v

  
if __name__ == "__main__":
  main()
