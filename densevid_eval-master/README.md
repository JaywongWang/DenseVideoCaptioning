# Dense Captioning Events in Video - Evaluation Code

## Usage
First, clone this repository and make sure that all the submodules are also cloned properly.
```
git clone --recursive https://github.com/ranjaykrishna/densevid_eval.git
```

Next, download the dataset using
```
./download.sh
```

Finally, test that the evaluator runs by testing it on our ```sample_submission.json``` file by calling:
```
python evaluate.py
```

You are now all set to produce your own dense captioning results for videos and use this code to evaluate your mode:
```
python evaluate.py -s YOUR_SUBMISSION_FILE.JSON
```

## Paper
Visit [our project page](http://cs.stanford.edu/people/ranjaykrishna/densevid) and read our paper for details.

## Citation
```
@inproceedings{krishna2017dense,
    title={Dense-Captioning Events in Videos},
    author={Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
    booktitle={ArXiv},
    year={2017}
}
```

## License

MIT License copyright Ranjay Krishna

## Contributions
Feel free to send pull requests to help write example code, contribute patches, document methods, etc. You can reach me through:

twitter: @RanjayKrishna

email: ranjaykrishna [at] stanford [dot] edu
