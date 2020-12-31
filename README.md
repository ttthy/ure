# Revisiting Unsupervised Relation Extraction

Source code for [Revisiting Unsupervised Relation Extraction](https://www.aclweb.org/anthology/2020.acl-main.669) in ACL 2020

## Environment

```bash
pip3 install -r requirements.txt
```
The experiments were conducted on Nvidia V100 GPUs (16GB GPU RAM).
However, these methods are very small, you can run on most GPU.

## Datasets
NYT: contact [Diego Marcheggiani](https://diegma.github.io/)
TACRED: [TACRED](https://nlp.stanford.edu/projects/tacred/)
Input format: same as [sample](https://github.com/diegma/relation-autoencoder/blob/master/data-sample.txt)

- Both NYT and TACRED are pre-processed (tokenisation, entity typing).
- We use Stanford CoreNLP to get dependency features for TACRED. 
- Entity types in NYT is a subset of TACRED, we map all entity types in TACRED that are unseen in NYT to `MISC`.

There are some vocabulary files needed to generate in advance.
You can use the script
```
bash ure/preprocessing/run.sh
```

We also provide the file for feature extraction
```
python ure/preprocessing/feature_extractor.py --input_file [file] --lexicon_file [file] --output_file [file] --threshold [occurrence threshold]
```


## Usage


### Training
EType+: B3 usually achieves 41% after one epoch 
```
python -u -m ure.etypeplus.main  --config models/etypeplus.yml
```

Feature [Marcheggiani and Titov](www.aclweb.org/anthology/Q16-1017.pdf): expect to get B3 around 32-33% after one epoch
```
python -u -m ure.feature.main --config models/feature.yml
```

PCNN [Simon et al](https://www.aclweb.org/anthology/P19-1133.pdf)
```
python -u -m ure.pcnn.main --config models/pcnn.yml
```

### Evaluation
```
python -u -m ure.etypeplus.main   --config models/etypeplus.yml --mode test
```


## Reproducibility & Bug Fixes & FQA

**L_s coefficient**
`rel_dist.py` is now shared among three methods in which `loss_s` is scaled down by `[B x k_samples]`, hence, the coefficient of `L_s` of `EType+` is set to `0.01` instead of `0.0001` in the paper.
(Line 91 in [/ure/rel_dist.py](https://github.com/ttthy/ure/ure/rel_dist.py))

**Entity type dimension** in Table 4. (b,c) appendix
There is a mistake, it is **entity dimension** in link predictor, we use the same dimension of `10` for all methods.
(There is no entity type in PCNN.)

**Typos in the paper**
Appendix A., in the second paragraph, the number of relation labels in NYT-FB should be 262 (253 in the paper).
Same for the caption of Figure 2a, NYT-FB has 262 relation types in total.
The last x axis label of Figure 2a. should be "each of the rest 249 relation types".


## Citation
If you plan to use it, please cite the following paper =)

```
@inproceedings{tran-etal-2020-revisiting,
    title = "Revisiting Unsupervised Relation Extraction",
    author = "Tran, Thy Thy  and
      Le, Phong  and
      Ananiadou, Sophia",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.669",
    pages = "7498--7505"
}
```
