import collections
import json
import os

import conll_alternates
import conll_lib
import convert_lib


def add_sentence(curr_doc, curr_sent, doc_coref_map, doc_parse_map,
                 sentence_offset):
  sequences = conll_lib.get_sequences(curr_sent, conll_lib.CONLL_FIELD_MAP)
  curr_doc.speakers.append(sequences[conll_lib.LabelSequences.SPEAKER])
  curr_doc.sentences.append(sequences[conll_lib.LabelSequences.WORD])
  #curr_doc.pos.append(sequences[conll_lib.LabelSequences.POS])

  coref_span_map = conll_lib.build_coref_span_map(
      sequences[conll_lib.LabelSequences.COREF], sentence_offset)
  doc_coref_map = conll_lib.ldd_append(doc_coref_map, coref_span_map)

  parse_span_map = conll_lib.build_parse_span_map(
      sequences[conll_lib.LabelSequences.PARSE], sentence_offset)
  doc_parse_map = conll_lib.ldd_append(doc_parse_map, parse_span_map)
  
  sentence_offset += len(sequences[conll_lib.LabelSequences.WORD])

  return doc_coref_map, doc_parse_map, sentence_offset#, sequences["POS"]


def create_dataset(filename, dataset_name):
 
  dataset = convert_lib.Dataset(dataset_name)

  list_data = conll_lib.listify_conll_dataset(filename)

  document_counter = 0

  for doc in list_data:
    sentence_offset = 0
    doc_coref_map = collections.defaultdict(list)
    doc_parse_map = collections.defaultdict(list)
    pos_sequences = []

    # Get document metadata from begin line
    begin_line = doc[0][0]
    assert begin_line[0] == "#begin"
    curr_doc_id = begin_line[2][1:-2]
    curr_doc_part = begin_line[-1]

    curr_doc = convert_lib.CorefDocument(
        curr_doc_id, curr_doc_part, init_status=convert_lib.ProcessingStage.TOKENIZED)

    sentences = doc[1:-1] # Excluding the #begin and #end lines
    for sentence in sentences:
      doc_coref_map, doc_parse_map, sentence_offset = add_sentence(
        curr_doc, sentence, doc_coref_map, doc_parse_map, sentence_offset)
      #pos_sequences.append(pos_seq)

    true_clusters = [clusters
        for key, clusters in doc_coref_map.items() if not key.startswith("s")]
  
    additional_mentions = sum([clusters
        for key, clusters in doc_coref_map.items() if key.startswith("s")], [])

    other_info = {"parse_map": [(k, v) for k, v in doc_coref_map.items()]}
    curr_doc.other_info_json = json.dumps(other_info)
  
    curr_doc.clusters = true_clusters
    curr_doc.additional_mentions = additional_mentions
    dataset.documents[convert_lib.ProcessingStage.TOKENIZED].append(curr_doc)

  if dataset_name == "classic":
    dataset.dump_to_conll(filename.replace(".txt", ".conll"),
                          drop_singletons=True)
    
  return dataset

def create_alternate_subdataset(data_home, original_dataset, new_dataset):
  input_directory = os.path.join(
      data_home, "original", original_dataset)
  output_directory = os.path.join(data_home, "original/conll_alternates", new_dataset)
  convert_lib.create_dir(output_directory)
    
  fn = conll_alternates.FN_MAP[new_dataset]

  for split in convert_lib.DatasetSplit.ALL:
    input_filename = os.path.join(input_directory, split + ".txt")
    output_filename = os.path.join(
                          output_directory, split + ".txt")
    dataset = conll_lib.listify_conll_dataset(input_filename)
    converted_dataset = conll_alternates.conll_add_singletons(dataset, fn)
    conll_alternates.write_conll_to_file(converted_dataset, output_filename) 


def convert_subdataset(data_home, dataset_name):
  input_directory = os.path.join(data_home, "original/conll_alternates", dataset_name)
  output_directory = os.path.join(data_home, "processed/conll", dataset_name)
  convert_lib.create_dir(output_directory)
  conll_datasets = {}
  for split in convert_lib.DatasetSplit.ALL:
    input_filename = os.path.join(input_directory, split + ".txt")
    converted_dataset = create_dataset(input_filename, dataset_name)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
 

def convert(data_home):
  """This just creates the alternate conlls, then converts everything."""
  alternate_subdatasets = [convert_lib.Variation.classic,
    convert_lib.Variation.sing, convert_lib.Variation.gold,
    convert_lib.Variation.goldconst]

  for subdataset in alternate_subdatasets:
    create_alternate_subdataset(
        data_home, convert_lib.DatasetName.conll, subdataset)
    convert_subdataset(data_home, subdataset)
