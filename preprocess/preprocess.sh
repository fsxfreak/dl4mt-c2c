#!/bin/bash

#example cmd:
#./preprocess.sh de en europarl-v7 /nfs/topaz/lcheung/data/europarl-de-en

# source language (example: de)
S=$1
# target language (example: en)
T=$2

PREFIX=$3

DATA_DIR=$4

# path to subword NMT scripts (can be downloaded from https://github.com/rsennrich/subword-nmt)
#P2=$4

## merge all parallel corpora
#./merge.sh $1 $2

date
echo "Preprocessing data for char2char model..."

perl normalize-punctuation.perl -l ${S} < ${DATA_DIR}/${PREFIX}.${S}-${T}.${S} > ${DATA_DIR}/${PREFIX}.${S}-${T}.${S}.norm  # do this for validation and test
perl normalize-punctuation.perl -l ${T} < ${DATA_DIR}/${PREFIX}.${S}-${T}.${T} > ${DATA_DIR}/${PREFIX}.${S}-${T}.${T}.norm  # do this for validation and test

# tokenize
echo "Finished normalizing punctuation, now tokenizing..."
perl tokenizer_apos.perl -threads 5 -l $S < ${DATA_DIR}/${PREFIX}.${S}-${T}.${S}.norm > ${DATA_DIR}/${PREFIX}.${S}-${T}.${S}.tok  # do this for validation and test
perl tokenizer_apos.perl -threads 5 -l $T < ${DATA_DIR}/${PREFIX}.${S}-${T}.${T}.norm > ${DATA_DIR}/${PREFIX}.${S}-${T}.${T}.tok  # do this for validation and test

# BPE
#if [ ! -f "../${S}.bpe" ]; then
#    python $P2/learn_bpe.py -s 20000 < all_${S}-${T}.${S}.tok > ../${S}.bpe
#fi
#if [ ! -f "../${T}.bpe" ]; then
#    python $P2/learn_bpe.py -s 20000 < all_${S}-${T}.${T}.tok > ../${T}.bpe
#fi

#python $P2/apply_bpe.py -c ../${S}.bpe < all_${S}-${T}.${S}.tok > all_${S}-${T}.${S}.tok.bpe  # do this for validation and test
#python $P2/apply_bpe.py -c ../${T}.bpe < all_${S}-${T}.${T}.tok > all_${S}-${T}.${T}.tok.bpe  # do this for validation and test

# shuffle 
#python $P1/shuffle.py all_${S}-${T}.${S}.tok.bpe all_${S}-${T}.${T}.tok.bpe all_${S}-${T}.${S}.tok all_${S}-${T}.${T}.tok

# build dictionary
echo "Now building dictionary"
python build_dictionary_char.py ${DATA_DIR}/${PREFIX}.${S}-${T}.${S}.tok 1000 1
python build_dictionary_char.py ${DATA_DIR}/${PREFIX}.${S}-${T}.${T}.tok 1000 1

date
echo "Finished preprocessing data for char2char model."
