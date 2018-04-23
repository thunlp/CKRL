# CKRL
Does William Shakespeare REALLY Write Hamlet? Knowledge Representation Learning with Confidence (AAAI-2018)

New: Add src


# INTRODUCTION

Confidence-aware Knowledge Representation Learning (CKRL)

Does William Shakespeare REALLY Write Hamlet? Knowledge Representation Learning with Confidence (AAAI-2018)

Written by Ruobing Xie


# DATA

FB15k is published by the author of the paper "Translating Embeddings for Modeling Multi-relational Data (2013)." 
<a href="https://everest.hds.utc.fr/doku.php?id=en:transe">[download]</a>
You can also get FB15k from here: <a href="http://pan.baidu.com/s/1eSvyY46">[download]</a>

data.zip contains the original FB15k dataset as well as 3 noisy datasets FB15K-N1, FB15K-N2, FB15K-N3 


# COMPILE 

Just type make in the folder ./


# RUN

train: ./Train_transC -size 50 -margin 1 -method 1

test: ./Test_transC bern


# CITE

If the codes or datasets help you, please cite the following paper:

Ruobing Xie, Zhiyuan Liu, Fen Lin, Leyu Lin. Does William Shakespeare REALLY Write Hamlet? Knowledge Representation Learning with Confidence (AAAI-2018)
