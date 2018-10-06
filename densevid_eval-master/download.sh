# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna 
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# -------------------------------------------------------- 

mkdir -p data/

echo '| Downloading the ActivityNet Captions dataset...'
wget http://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip 
unzip captions
mv -t data train_ids.json val_ids.json test_ids.json train.json val_1.json val_2.json readme.txt
rm captions.zip

echo "| done."
