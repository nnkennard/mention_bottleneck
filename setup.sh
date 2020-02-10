MODEL_DIR_PATH=$1
ONTONOTES_PATH=$2
DATA_PATH=./data

# Clone necessary model repositories into MODEL_DIR_PATH 
# BERS
git clone https://github.com/gregdurrett/berkeley-entity.git $1/berkeley-entity
wget http://nlp.cs.berkeley.edu/downloads/berkeley-entity-models.tgz
mv berkeley-entity-models.tgz ../mention_bottleneck_models/berkeley-entity/

wget http://nlp.cs.berkeley.edu/downloads/berkeleycoref-1.1.tgz
wget http://nlp.cs.berkeley.edu/downloads/berkeleycoref-1.0-models.tgz
tar -xvzf berkeleycoref-1.0-models.tgz
tar -xvzf berkeleycoref-1.1.tgz
mv models/ berkeleycoref
rm berkeleycoref-1.0-models.tgz
rm berkeleycoref-1.1.tgz

# SpanBERT coref
# Berkeley Coreference Analyzer
# e2ecoref

# Get and preprocess data
mkdir data

# Wikicoref
wget http://rali.iro.umontreal.ca/rali/sites/default/files/resources/wikicoref/WikiCoref.tar.gz
tar -zxf WikiCoref.tar.gz
mv WikiCoref/ data/

python convert_all_datasets.py data/

# Ontonotes experiments

# Run basic and "gold" experiments

java -cp ./*.jar -Xmx10g edu.berkeley.nlp.coref.preprocess.PreprocessingDriver ++base.conf \
  -execDir exec_dir \
  -inputDir ~/mention_bottleneck/data/processed/wikicoref/test-fpd/ \
  -outputDir preprocessed/wikicoref \

java -jar -Xmx20g berkeleycoref-1.1.jar ++base.conf \
-execDir exec_dir \
-modelPath models/coref-rawtext-final.ser \
-outputPath ./outputs/wikicoref \
-testPath preprocessed/wikicoref/ \
-numberGenderData gender.data \
-mode PREDICT -docSuffix txt

java -jar -Xmx20g berkeleycoref-1.1.jar ++base.conf \
-execDir exec_dir \
-modelPath models/coref-rawtext-final.ser \
-outputPath ./gold_outputs/wikicoref \
-testPath preprocessed/wikicoref/ \
-numberGenderData gender.data \
-mode PREDICT -docSuffix txt \
-useGoldMentions


# Calculate recall

# Run BCA

# Out of domain experiments
