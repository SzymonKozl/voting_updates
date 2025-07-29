# Updating NN weights with proportional votes aggregations
General goal: explore effectiveness of updating the weights of NN by electing groups of weights that will be incremented and decremented.
In particular, we are interested in utilizing MES in this process.

## Sources
* https://ojs.aaai.org/index.php/AAAI/article/view/28813 - 6.3 - interesting perspective for possible application (we could create combined model by using the new technique)
* https://www.arxiv.org/pdf/2503.01985 p- theoretical background for negative votes aggregation


## Milestones
* general evaluation framework
* bos/mes implementation
* dataset that represents the problem
* tests

## Update 1: 29.07.25
Initial experiments indicates extremally high computational cost (as expected) of calculating MES. Considered next steps:
* split weights into smaller packs (not several times bigger than num of voters and do MES on each pack)
* try to approximate MES
* use other rule
* apply MES only to selected layers

| voters (batch size) | candidates (num of weights * 2) | time (ms) |
|---------------------|---------------------------------|-----------|
| 32                  | 10                              | 0.58      |
| 32                  | 100                             | 9.96      |
| 32                  | 1000                            | 592.88    |
| 32                  | 10000                           | 62346.97  |
tab 1: mean time for computing binary decision MES  (10 trials)