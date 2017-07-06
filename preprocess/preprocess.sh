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


SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
cd "$SCRIPT_DIR"

date
echo "Preprocessing data for char2char model..."

BASE_SRC=${DATA_DIR}/${PREFIX}.${S}-${T}.${S}
BASE_TRG=${DATA_DIR}/${PREFIX}.${S}-${T}.${T}

#python3 split_data.py /nfs/topaz/lcheung/data/europarl-de-en/europarl-v7.de-en.en /nfs/topaz/lcheung/data/europarl-de-en/e 
python3 split_data.py ${BASE_SRC} ${BASE_TRG}

BASE_SRC_TRAIN=${BASE_SRC}.train
BASE_SRC_DEV=${BASE_SRC}.dev
BASE_SRC_TEST=${BASE_SRC}.test
BASE_TRG_TRAIN=${BASE_TRG}.train
BASE_TRG_DEV=${BASE_TRG}.dev
BASE_TRG_TEST=${BASE_TRG}.test

perl normalize-punctuation.perl -l ${S} < ${BASE_SRC_TRAIN} > ${BASE_SRC_TRAIN}.norm 
perl normalize-punctuation.perl -l ${S} < ${BASE_SRC_DEV} > ${BASE_SRC_DEV}.norm 
perl normalize-punctuation.perl -l ${S} < ${BASE_SRC_TEST} > ${BASE_SRC_TEST}.norm 
perl normalize-punctuation.perl -l ${T} < ${BASE_TRG_TRAIN} > ${BASE_TRG_TRAIN}.norm  
perl normalize-punctuation.perl -l ${T} < ${BASE_TRG_DEV} > ${BASE_TRG_DEV}.norm  
perl normalize-punctuation.perl -l ${T} < ${BASE_TRG_TEST} > ${BASE_TRG_TEST}.norm  

rm $BASE_SRC_TRAIN
rm $BASE_SRC_DEV
rm $BASE_SRC_TEST
rm $BASE_TRG_TRAIN
rm $BASE_TRG_DEV
rm $BASE_TRG_TEST

BASE_SRC_TRAIN=${BASE_SRC_TRAIN}.norm
BASE_SRC_DEV=${BASE_SRC_DEV}.norm
BASE_SRC_TEST=${BASE_SRC_TEST}.norm
BASE_TRG_TRAIN=${BASE_TRG_TRAIN}.norm
BASE_TRG_DEV=${BASE_TRG_DEV}.norm
BASE_TRG_TEST=${BASE_TRG_TEST}.norm

# tokenize
echo "Finished normalizing punctuation, now tokenizing..."
perl tokenizer_apos.perl -threads 9 -l $S < ${BASE_SRC_TRAIN} > ${BASE_SRC_TRAIN}.tok 
perl tokenizer_apos.perl -threads 9 -l $S < ${BASE_SRC_DEV} > ${BASE_SRC_DEV}.tok 
perl tokenizer_apos.perl -threads 9 -l $S < ${BASE_SRC_TEST} > ${BASE_SRC_TEST}.tok 
perl tokenizer_apos.perl -threads 9 -l $S < ${BASE_TRG_TRAIN} > ${BASE_TRG_TRAIN}.tok 
perl tokenizer_apos.perl -threads 9 -l $S < ${BASE_TRG_DEV} > ${BASE_TRG_DEV}.tok 
perl tokenizer_apos.perl -threads 9 -l $S < ${BASE_TRG_TEST} > ${BASE_TRG_TEST}.tok 

rm $BASE_SRC_TRAIN
rm $BASE_SRC_DEV
rm $BASE_SRC_TEST
rm $BASE_TRG_TRAIN
rm $BASE_TRG_DEV
rm $BASE_TRG_TEST

BASE_SRC_TRAIN=${BASE_SRC_TRAIN}.tok
BASE_SRC_DEV=${BASE_SRC_DEV}.tok
BASE_SRC_TEST=${BASE_SRC_TEST}.tok
BASE_TRG_TRAIN=${BASE_TRG_TRAIN}.tok
BASE_TRG_DEV=${BASE_TRG_DEV}.tok
BASE_TRG_TEST=${BASE_TRG_TEST}.tok

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
cat ${BASE_SRC_TRAIN} ${BASE_SRC_DEV} ${BASE_SRC_TEST} > ${BASE_SRC}.tok
python build_dictionary_char.py ${BASE_TRG}.tok 1000 1
rm ${BASE_SRC}.tok
cat ${BASE_TRG_TRAIN} ${BASE_TRG_DEV} ${BASE_TRG_TEST} > ${BASE_TRG}.tok
python build_dictionary_char.py ${BASE_SRC}.tok 1000 1
rm ${BASE_TRG}.tok

date
echo "Finished preprocessing data for char2char model."
