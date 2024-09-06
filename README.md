# ReferenceInstancesPredictability

Code for the paper [100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances](https://arxiv.org/abs/2409.03563) 


## How to reproduce results
### Install dependencies

Most of the dependencies can be installed by running
```bash
pip install -r requirements.txt
``` 

In order to run the IRT part, the following must be done:
- install pyro: `pip3 install pyro-ppl==1.8.6`
- install my fork of py-irt: `pip3 install git+https://github.com/LoryPack/py-irt.git`. This is because a small adaptation had to be done to the `py-irt` library. My fork is an adaptation of `https://github.com/felipemaiapolo/py-irt`, which however caused some dependency issues.


### Run the experiments
Steps: 
1. get the raw data
  - KindsOfReasoning: download this file from [this repo](https://github.com/Kinds-of-Intelligence-CFI/KindsOfReasoning/tree/main/full_processing_steps) and decompress it in a folder `results/kindsofreasoning` in the root of this repository 
    - HELM-Lite: download the HELM data by running the `download_lite.ipynb` notebook in `experiments/download_helm`. This downloads all the necessary files in `results/helm_lite_v1.0.0`. Notice that this takes long (3.6GB).
2. compute the embeddings running the two scripts `experiments/0_run_openai_embeddings_all_kindsofreasoning.py` and `experiments/0_run_openai_embeddings_all_helm.py`, which will create two new folders in `results` where the computed embeddings will be stored. These require an OpenAI API key to be set in `.env`. Computing the embeddings is a bit slow but cheap; unfortunately the resulting files are large so they cannot be easily stored on GitHub.
3. Run the experiments by running the various notebooks in the `experiments` folder. They will create two subfolders (`results` and `fig`) where the result files and figures will be stored.


## Citation

If you use our code, please cite our paper using the following: 

```bibtex
@misc{pacchiardi2024100instancesneedpredicting,
      title={100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances}, 
      author={Lorenzo Pacchiardi and Lucy G. Cheke and José Hernández-Orallo},
      year={2024},
      eprint={2409.03563},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.03563}, 
}
```


# Credits

- The code for running the IRT model was adapted from [this repository](https://github.com/felipemaiapolo/tinyBenchmarks), released under MIT License.
- The code to download HELM-Lite was adapted from [this file](https://github.com/felipemaiapolo/efficbench/tree/master/generating_data/download_helm).
- The code to compute Word2Vec and FastText embeddings was adapted from https://github.com/lorypack/llm-liedetector (released under BSD-3-Clause license)
