# Reference implementation for [Diffusion LLMs are natural adversaries for black-box LLMs](https://arxiv.org/abs/2511.00203)

## Run Inpainting attack via Adversarial LLM toolkit
[https://github.com/LLM-QC/AdversariaLLM/tree/feat/add_inpainting_attack](https://github.com/LLM-QC/AdversariaLLM)

## Code

### Installation
Conda environment
```python
conda create -n llm-inpainting-attack python=3.11 -y
conda activate llm-inpainting-attack
```

hf_home environment variable
```python
export HF_HOME=/path/to/huggingface/cache
```

Using Python 3.11.11
```
pip install -r requirements.txt
```

### Attack
To generate new adversarial prompts, you may execute follow the `generate.slurm` file.

You can also pass a path to a CSV file for custom prompts and targets. See [csv example](./csv_example.csv) for structure.
```python
python generate.py --dataset ./csv_example.csv
```
This creates (in default settings) a CSV file in the output folder containing the generated adversarial prompts.

## Citation
@article{ludke2025diffusion,
  title={Diffusion LLMs are Natural Adversaries for any LLM},
  author={L{\"u}dke, David and Wollschl{\"a}ger, Tom and Ungermann, Paul and G{\"u}nnemann, Stephan and Schwinn, Leo},
  journal={arXiv preprint arXiv:2511.00203},
  year={2025}
}
