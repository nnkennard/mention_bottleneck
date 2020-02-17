import conll_lib
import convert_lib
import collections
import os
import re

CONLL = convert_lib.DatasetName.conll

def add_sentence(curr_doc, curr_sent, doc_coref_map, doc_parse_map,
                 sentence_offset):
  sequences = conll_lib.get_sequences(curr_sent, conll_lib.CONLL_FIELD_MAP)
  curr_doc.speakers.append(sequences[conll_lib.LabelSequences.SPEAKER])
  curr_doc.sentences.append(sequences[conll_lib.LabelSequences.WORD])
  curr_doc.pos.append(sequences[conll_lib.LabelSequences.POS])

  coref_span_map = conll_lib.build_coref_span_map(
      sequences[conll_lib.LabelSequences.COREF], sentence_offset)
  doc_coref_map = conll_lib.ldd_append(doc_coref_map, coref_span_map)

  parse_span_map = conll_lib.build_parse_span_map(
      sequences[conll_lib.LabelSequences.PARSE], sentence_offset)
  doc_parse_map = conll_lib.ldd_append(doc_parse_map, parse_span_map)
  
  sentence_offset += len(sequences[conll_lib.LabelSequences.WORD])

  return doc_coref_map, doc_parse_map, sentence_offset


def create_dataset(filename):
 
  dataset = convert_lib.Dataset(CONLL)

  list_data = conll_lib.listify_conll_dataset(filename)

  document_counter = 0
  sentence_offset = 0

  for doc in list_data:
    doc_coref_map = collections.defaultdict(list)
    doc_parse_map = collections.defaultdict(list)
    begin_line = doc[0][0]
    assert begin_line[0] == "#begin"
    curr_doc_id = begin_line[2][1:-2].replace("/", "-")
    curr_doc_part = begin_line[-1]
    curr_doc = convert_lib.Document(curr_doc_id, curr_doc_part)
    sentences = doc[1:-1] # Excluding the #begin and #end lines
    for sentence in sentences:
      doc_coref_map, doc_parse_map, sentence_offset = add_sentence(
        curr_doc, sentence, doc_coref_map, doc_parse_map, sentence_offset)
    dataset.documents.append(curr_doc)
    
  return dataset


def convert(data_home):
  input_directory = os.path.join(data_home, "original", CONLL)
  output_directory = os.path.join(data_home, "processed", CONLL)
  convert_lib.create_dir(output_directory)
  conll_datasets = {}
  for split in convert_lib.DatasetSplit.ALL:
    input_filename = os.path.join(input_directory, "conll12_" + split + ".txt")
    converted_dataset = create_dataset(input_filename)
    convert_lib.write_converted(converted_dataset, output_directory + "/" + split)
