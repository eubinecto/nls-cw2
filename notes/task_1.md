# Named Entity Recognition

## Use the function `nltk.ne_chunk()` to process corpus 1

> what is "corpus_1"?

From Blackboard:
```text
Task 1:
Corpus 1:  US presidential inaugural addresses 1789-2009 
(Also available at https://archive.org/details/Inaugural-Address-Corpus-1789-2009)
```


> how do I use `nltk.ne_chunk()`?

You can find the documentation from [here](http://www.nltk.org/book/ch07.html).

Here is an exmple snippet for this:

```python
import nltk
# rule-based ner.
sent = nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent, binary=True)) [1]
```
```commandline
(S
  The/DT
  (NE U.S./NNP)
  is/VBZ
  one/CD
  ...
  according/VBG
  to/TO
  (NE Brooke/NNP T./NNP Mossman/NNP)
  ...
```



## Use the Stanford named-entity recogniser to process the same corpus

> Now how do I use that?

nltk provides an interface to Stanford's NER.
- http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford


 
## Compare the outputs of the two NER methods for the ORGANIZATION class.

So, what matters is the analysis, right?


