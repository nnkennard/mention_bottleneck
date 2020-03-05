"""Resplit preco to make a test set, and create a non-singleton version."""

import json
import sys
import os
import random

import convert_lib

def get_records_from_preco_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()

def remove_singletons(records):
  new_records = []
  for record in records:
    obj = json.loads(record)
    new_clusters = []
    for cluster in obj["mention_clusters"]:
      if len(cluster) > 1:
        new_clusters.append(cluster)
    obj["mention_clusters"] = new_clusters
    new_records.append(json.dumps(obj))
  return new_records
  

def preprocess(data_dir):

  preco_orig_dir = os.path.join(data_dir, "original", "PreCo_1.0")
  preco_dir = os.path.join(data_dir, "original", "preco_sing")
  preco_mult_dir = os.path.join(data_dir, "original", "preco_mult")
  
  convert_lib.create_dir(preco_dir)
  convert_lib.create_dir(preco_mult_dir)

  resplit_datasets = {}

  resplit_datasets[
      convert_lib.DatasetSplit.test] = get_records_from_preco_file(
      os.path.join(preco_orig_dir, "dev.jsonl"))

  temp_original_train = get_records_from_preco_file(
      os.path.join(preco_orig_dir, "train.jsonl"))
  random.seed(43)
  random.shuffle(temp_original_train)
  total_train = len(temp_original_train)
  boundary = int(0.8 * total_train)
  resplit_datasets[
      convert_lib.DatasetSplit.train] = temp_original_train[:boundary]
  resplit_datasets[
      convert_lib.DatasetSplit.dev] = temp_original_train[boundary:]


  for split_name, records in resplit_datasets.items():
    # Write resplit
    with open(os.path.join(preco_dir, split_name + ".jsonl"), 'w') as f:
      f.write("".join(records))
    # Write resplit mult
    no_singleton_records = remove_singletons(records)
    with open(os.path.join(preco_mult_dir, split_name + ".jsonl"), 'w') as f:
      f.write("\n".join(no_singleton_records))
  
