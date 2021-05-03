# NLS CW2
- author: Eu-Bin KIM
- date of submission: 4th of May 2021


## Task 1 (10 marks) - should be just 1-page long.
### Processing the Inaugural corpus

> this should be only 

### Comparing and Discussions
#### Boundary detection discussion

- nltk's method - quite a lot of false-positives.
  - e.g. at line 008: "(S\n  In/IN\n  tendering/VBG\n  this/DT\n  homage/NN\n  to/TO\n  the/DT\n  (ORGANIZATION Great/NNP Author/NNP)\n  of/IN\n  every/DT\n  public/NN\n  and/CC\n  private/JJ\n  good/JJ\n  ,/,\n  I/PRP\n  assure/VBP\n  myself/PRP\n  that/IN\n  it/PRP\n  expresses/VBZ\n  your/PRP$\n  sentiments/NNS\n  not/RB\n  less/RBR\n  than/IN\n  my/PRP$\n  own/JJ\n  ,/,\n  nor/CC\n  those/DT\n  of/IN\n  my/PRP$\n  fellow/JJ\n  citizens/NNS\n  at/IN\n  large/JJ\n  less/JJR\n  than/IN\n  either/DT\n  ./.)"
  - e.g. at line 250: "(S\n  How/WRB\n  did/VBD\n  we/PRP\n  accomplish/VB\n  the/DT\n  (ORGANIZATION Revolution/NNP)\n  ?/.)"
  - e.g. at line 182:  "(S\n  The/DT\n  proofs/NN\n  are/VBP\n  in/IN\n  the/DT\n  records/NNS\n  of/IN\n  each/DT\n  successive/JJ\n  (ORGANIZATION Administration/NN)\n  of/IN\n  our/PRP$\n  Government/NNP\n  ,/,\n  and/CC\n  the/DT\n  cruel/JJ\n  sufferings/NNS\n  of/IN\n  that/DT\n  portion/NN\n  of/IN\n  the/DT\n  (GPE American/JJ)\n  people/NNS\n  have/VBP\n  found/VBN\n  their/PRP$\n  way/NN\n  to/TO\n  every/DT\n  bosom/NN\n  not/RB\n  dead/JJ\n  to/TO\n  the/DT\n  sympathies/NNS\n  of/IN\n  human/JJ\n  nature/NN\n  ./.)"
- haven't checked it yet, but I'm pretty sure the result of Stanford's is better than this. Do put this into a table, alright?


####  Agreement between the tools

> exact match?
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
