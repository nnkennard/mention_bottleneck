import collections
import json
import sys

def e2e_pred_set(tokens):

  messy_counter = 0
  for sentence in tokens:
    for start in range(len(sentence)):
      for end in range(start, len(sentence)):
        if end - start < 30:
          messy_counter += 1
  return len(sum(tokens, [])) * 30, messy_counter

def check_document(doc_obj):
  inject_mentions = set(tuple(i) for i in doc_obj["inject_mentions"])
  gold_mentions = set(tuple(i) for i in sum(doc_obj["clusters"], []))

  tp = inject_mentions.intersection(gold_mentions)
  fp = inject_mentions - gold_mentions
  fn = gold_mentions - inject_mentions
  num_e2e_spans, messy_counter = e2e_pred_set(doc_obj["sentences"])

  print("\t".join(
      str(len(i)) for i in [gold_mentions, tp, fp, fn]) + "\t" + str(num_e2e_spans) + "\t" + str(messy_counter))

  return tp, fp, fn



def main():
  input_jsonl_file = sys.argv[1]

  problem_collector = collections.Counter()
  
  with open(input_jsonl_file, 'r') as f:
    for line in f:
      doc_obj = json.loads(line)
      tp, fp, fn = check_document(doc_obj)
      problem_collector["tp"] += len(tp)
      problem_collector["fp"] += len(fp)
      problem_collector["fn"] += len(fn)

  print(problem_collector)
  

if __name__ == "__main__":
  main()
