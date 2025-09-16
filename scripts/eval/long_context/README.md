# Evaluation with `long_context`

`long_context` evaluation uses the [BABILong](https://github.com/booydar/babilong) package for needle-in-a-haystack style benchmarking. No installation is required—just clone the repository:

```bash
git clone https://github.com/booydar/babilong source
```

After cloning BABILong, you can evaluate a model using the [`eval_long_context.py`](./eval_long_context.py) script with different test-time context extension options:

```bash
python eval_long_context.py --model_name microsoft/phi-4 --rope_scaling_type none
python eval_long_context.py --model_name microsoft/phi-4 --rope_scaling_type linear
python eval_long_context.py --model_name microsoft/phi-4 --rope_theta 1000000
python eval_long_context.py --model_name Qwen/Qwen2.5-14B-Instruct --rope_scaling_type yarn_qwen
python eval_long_context.py --model_name meta-llama/Llama-3.1-8B-Instruct
```

To visualize results, use the [`plot_eval_long_context.ipynb`](./plot_eval_long_context.ipynb) notebook provided in the repository.

*For more information, please refer to this [notebook](https://github.com/booydar/babilong/blob/main/notebooks/babilong_evaluation_example.ipynb).*