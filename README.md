Fully Character-Level Neural Machine Translation
==================================

Theano implementation of the models described in the paper [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/abs/1610.03017 "Fully Character-Level Neural Machine Translation without Explicit Segmentation").

We present code for training and decoding four different models:

1. bilingual bpe2char (from [Chung et al., 2016](https://arxiv.org/abs/1603.06147)).
2. bilingual char2char
3. multilingual bpe2char
4. multilingual char2char

This fork has extra code for forced decoding, and easier-to-use parameters for custom datasets (with support currently only for bilingual char2char).

Dependencies
------------------
### Python
* Theano
* Numpy
* NLTK

### GPU
* CUDA (we recommend using the latest version. The version 8.0 was used in all our experiments.)

### Related code
* For preprocessing and evaluation, we used scripts from [MOSES](https://github.com/moses-smt/mosesdecoder "MOSES").
* This code is based on [Subword-NMT](http://arxiv.org/abs/1508.07909 "Subword-NMT") and [dl4mt-cdec](https://github.com/nyu-dl/dl4mt-cdec "dl4mt-cdec").

Downloading Datasets & Pre-trained Models
------------------
The original WMT'15 corpora can be downloaded from [here](http://www.statmt.org/wmt15/translation-task.html). For the preprocessed corpora used in our experiments, see below.
* WMT'15 preprocessed corpora
  * [Standard version (for bilingual models, 3.5GB)](https://drive.google.com/open?id=0BxmEQ91VZAPQam5pc2ltQ1BBTTQ)
  * [Cyrillic converted to Latin (for multilingual models, 2.6GB)](https://drive.google.com/open?id=0BxmEQ91VZAPQS0oxTDJINng5b1k)

To obtain the pre-trained top-performing models, see below.
* [Pre-trained models (6.0GB)](https://drive.google.com/open?id=0BxmEQ91VZAPQcGx4VGI2N3dMNEE): **Tarball updated** on Nov 21st 2016. The CS-EN bi-char2char model in the previous tarball was not the best-performing model. 

Training Details
------------------
### Using GPUs
Do the following before executing `train*.py`.
```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32
```
With space permitting on your GPU, it may speed up training to use `cnmem`:
```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.95,allow_gc=False
```

On a pre-2016 Titan X GPU with 12GB RAM, our bpe2char models were trained with `cnmem`. Our char2char models (both bilingual and multilingual) were trained without `cnmem` (due to lack of RAM).

#### Bilingual bpe2char
```bash
$ python bpe2char/train_bi_bpe2char.py -translate <LANGUAGE_PAIR>
```
#### Bilingual char2char
This is an example command, so make sure to change the parameters.
```bash
$ python char2char/train_bi_char2char.py \                       
        -src_train "$SRC_TRAIN" \                              
        -src_dev "$SRC_DEV" \                                  
        -trg_train "$TRG_TRAIN" \                              
        -trg_dev "$TRG_DEV" \                                  
        -src_dict "$SRC_DICT" \                                
        -trg_dict "$TRG_DICT" \                                
        -model_path /nfs/topaz/lcheung/models/dl4mt-toy-dict/ \
        -n_words_src 9 \                                       
        -n_words 9 \                                           
        -re_load -re_load_old_setting \                        
        -sampleFreq 2500 \                                     
        -saveFreq 1000 \                                       
        -dispFreq 500                                          
```
#### Multilingual bpe2char
```bash
$ python bpe2char/train_multi_bpe2char.py 
```
#### Multilingual char2char
```bash
$ python char2char/train_multi_char2char.py 
```
#### Checkpoint
To resume training a model from a checkpoint, simply append `-re_load` and `-re_load_old_setting` above. Make sure the checkpoint resides in the correct directory (`.../dl4mt-c2c/models`).

### Using Custom Datasets
To train your models using your own dataset (and not the WMT'15 corpus), you first need to learn your vocabulary using `build_dictionary_char.py` or `build_dictionary_word.py` for char2char or bpe2char model, respectively. For the bpe2char model, you additionally need to learn your BPE segmentation rules on the source corpus using the Subword-NMT repository (see below).

```bash
$ python build_dictionary_char.py src.train [word padding] 1
$ python build_dictionary_char.py trg.train [word padding] 1
```

Make sure to take the correct size of the dictionary from the output of the program.

Decoding
------------------

### Decoding validation / test files.

```bash
$ export THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.95,allow_gc=False
$ python code/dl4mt-c2c/translate/translate_char2char.py \
    -model "/nfs/topaz/lcheung/models/dl4mt-toy-dict/bi-char2char.grads.6000.npz" \                                  
    -saveto "output/dl-dec-bi-toy-dict-${num}" \
    -dict_src "src.train.404.pkl" \                            
    -dict_trg "trg.train.404.pkl" \                          
    -source "src.test.small" \                                 
```

When choosing which pre-trained model to give to `-model`, make sure to choose e.g. `.grads.123000.npz`. The models with `.grads.[num]` in their names are the optimal models and you should be decoding from those.

### Decoding an arbitrary file

If you choose to decode your own source file, make sure it is:

1. properly tokenized (using `preprocess/preprocess.sh`).
2. bpe-tokenized for bpe2char models.
3. Cyrillic characters should be converted to Latin for multilingual models.

### Decoding multilingual models
In progress.

Evaluation
------------------
We use the script from MOSES to compute the bleu score. The reference translations can be found in `.../wmt15`.
```
perl preprocess/multi-bleu.perl reference.txt < model_output.txt
```

Extra
-----------------
### Extracting & applying BPE rules

Clone the Subword-NMT repository.
```bash
git clone https://github.com/rsennrich/subword-nmt
```

Use following commands (find more information in [Subword-NMT](https://github.com/rsennrich/subword-nmt))
```bash
./learn_bpe.py -s {num_operations} < {train_file} > {codes_file}
./apply_bpe.py -c {codes_file} < {test_file}
```

### Converting Cyrillic to Latin

```bash
$ python preprocess/iso.py russian_source.txt
```
will produce an output at `russian_source.txt.iso9`.

Citation
------------------

```
@article{Lee:16,
  author    = {Jason Lee and Kyunghyun Cho and Thomas Hofmann},
  title     = {Fully Character-Level Neural Machine Translation without Explicit Segmentation},
  year      = {2016},
  journal   = {arXiv preprint arXiv:1610.03017},
}
```
