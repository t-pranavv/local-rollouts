# Evaluation with `evalchemy`

`evalchemy` is a package that provides a simple API for evaluating models on various tasks. It can be installed as follows:

```bash
git clone https://github.com/caiom/evalchemy.git
pip install -e evalchemy[eval]
pip install -e evalchemy/eval/chat_benchmarks/alpaca_eval
```

After installing `evalchemy`, you can evaluate a model as follows:

```bash
python eval_evalchemy.py --model_path microsoft/phi-4 --output_path <output_path> --num_evals 1 --num_concurrent 20 --work_dir evalchemy --wait 200 --temperature 0.1 --top_p 0.95 --system_prompt_type simple --tasks LiveCodeBench --max_gen_toks 14500
```

*For more information, please refer to the [evalchemy documentation](https://github.com/mlfoundations/evalchemy).*