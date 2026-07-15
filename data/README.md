# COUNTERFACT-Strict

`counterfact_50_strict.jsonl` is the final 50-example subset used in the report. It is derived from the 21,919-record [`NeelNanda/counterfact-tracing`](https://huggingface.co/datasets/NeelNanda/counterfact-tracing) adaptation of the CounterFact dataset introduced by Meng et al. (2022). The upstream ROME repository is distributed under the MIT License.

## Schema

Each JSONL record contains:

- `id`: stable source-derived identifier;
- `category` and `category_name`: relation identifier and readable relation;
- `subject` and `relation`: entities used to construct the strict QA prompt;
- `o_true`: factual answer;
- `o_false`: counterfactual target;
- `_original_prompt`: upstream completion template retained for provenance.

The subset includes only single-valued nominal answers, unambiguous subject-relation pairs, and distinct true/false objects. It is small and relation-skewed, so it should be treated as a mechanism-focused benchmark rather than a representative factuality sample.

## Regeneration

Regenerate the subset with:

```bash
python tools/prepare_counterfact.py
```

The Hugging Face `datasets` package and network access are required. Other downloaded or intermediate datasets in this directory are ignored by Git.

## Citation and Terms

Users should cite *Locating and Editing Factual Associations in GPT* (Meng et al., NeurIPS 2022) when using this subset. The repository's MIT License covers the original selection and transformation code; upstream data remains subject to its source terms and applicable law.
