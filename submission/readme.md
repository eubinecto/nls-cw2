# How to run the code

## 1.Unzip the archive
```shell
unzip delieverables.zip
```

## 2. enter the directory
```shell
cd delieverables
```
## 3. Install libraries 
```shell
pip3 install nltk
pip3 install scikit-learn
pip3 install gensim
```

## Task 1 - Named Entity Recognition
```shell
# process corpus 1 with nltk's ner
python3 -m nls_cw2.task_1.ner_with_nltk
# process corpus 1 with stanford's ner
python3 -m nls_cw2.task_1.ner_with_stan
# measure the agreements agreements
python3 -m nls_cw2.task_1.measure_agreements
```

## Task 2 - Sentiment Analysis

### part a
```shell
# collect new adjectives with the basic patter
python3 -m nls_cw2.task_2.part_a.collect_adjs_basic

# collect new adjectives with more patterns
python3 -m nls_cw2.task_2.part_a.collect_adjs_more

# assign polarities to the newly collected adjectives
python3 -m nls_cw2.task_2.part_a.assign_polarities

# evaluate the patterns
python3 -m nls_cw2.task_2.part_a.evaluate_patterns
```

### part b

```shell
# build the dataset
python3 -m nls_cw2.task_2.part_b.build_dataset

# evaluate a baseline model
python3 -m nls_cw2.task_2.part_b.evaluate_models --model="base"

# evaluate a bow model
python3 -m nls_cw2.task_2.part_b.evaluate_models --model="bow" --k=4

# evaluate a bow model with more features
python3 -m nls_cw2.task_2.part_b.evaluate_models --model="bow" --k=4 --more_features

# evaluate a w2v model
python3 -m nls_cw2.task_2.part_b.evaluate_models --model="w2v" --k=4

# evaluate a w2v model with more features
python3 -m nls_cw2.task_2.part_b.evaluate_models --model="w2v" --k=4 --more_features

```