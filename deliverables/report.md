# NLS CW2 - NER & Sentiment Analysis
- author: Eu-Bin KIM
- date of submission: 4th of May 2021

## Task 1 (should be just 1-page long)
### NER processing

The Inaugural corpus is ner-processed with NLTK NER in `nls_cw2/task_1/ner_with_nltk.py`, and with Stanford NER in
`nls_cw2/task_1/ner_with_stan.py`. The results of this process is saved in `data/task_1/ner_with_nltk.ndjson` and 
`data/task_1/ner_with_stan.djson`, respectively. 

### Comparing and Discussions

>line | NER-tagged with NLTK NER |  NER_tagged with Stanford NER
> --- | --- | --- 
> line 8 | In/IN tendering/VBG this/DT homage/NN\ to/TO the/DT **(ORGANIZATION Great/NNP Author/NNP)** of/IN ... | In/O  tendering/O, this/O, homage/O, to/O, the/O, **Great/O, Author/O**, of/O
> line 250 | How/WRB  did/VBD we/PRP accomplish/VB  the/DT **(ORGANIZATION Revolution/NNP)** ? | How/O  did/O we/O accomplish/O  the/O **Revolution\O** ?
> **Table 1**: Two examples of NER-tagged results but with different methods. 

**boundary detection discussion**. NLTK NER performs far worse than Stanford NER.
This is because we observe that NLTK NER tends to produce far more false-positives than Stanford NER does.
For instance, what Washington was referencing to with "the Great Author"(**Table 1**, line 8) was the jesus,
yet NLTK NER has falsely recognized it as an organization. Likewise, In the context of the sentence in line 250, what 
"Revolution" means is an achievement of some sort, yet NLTK NER has failed to recognize it correctly. In contrast,
Stanford NER has correctly recognized both of them as non-organizations.

> NLTK NER matches | Stanford NER matches | exact matches | partial matches
> --- | --- | --- | ---
> 890 | 770 | 389 | 124
> **Table 2**: Exact & partial matches of ORGANIZATION entities between the results of NLTK NER and Stanford NER.


**Agreement between tools*.
- CI
- why? - what patterns have you discovered in those that have been matched exactly?
- examples?
- explanations?
- wrap-up?


> partial overlap?
- CI
- why? - what patterns have you discovered in those that are partially overlapping?
- examples?
- explanations?
- wrap-up?


## Task 2 (30 marks)





- limits : e.g. 
  - wow . i have not been this disappointed by a movie in a long time . 


baseline | bow | bow with additional features | w2v | w2v with additional features
--- | --- | --- | --- | --- 
0.5139 | 0.6459 | 0.6475 | 0.6937 | 0.7009
