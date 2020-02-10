import sys
import wikicoref_converter
import preco_converter

def main():
  data_home = sys.argv[1]
  #print("Wikicoref")
  #wikicoref_converter.convert(data_home) 
  print("Preco")
  preco_converter.convert(data_home) 

if __name__ == "__main__":
  main()
