

def get_examples(filename):
  with open(filename, 'r') as f:
    return [json.loads(line) for line in f.readlines()]

def preco_gold(mult_filename):
  examples = get_examples(mult_filename)
  for example in examples:
    example["injected_mentions"] = sum(example["clusters"], [])
  out_file = mult_filename.replace("mult", "gold")
  with open(out_file, 'w') as f:
    f.write("\n".join(examples))


def preco_
