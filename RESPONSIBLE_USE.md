# Responsible Use and Release Policy

## Intended Use

SAST-IR is intended for authorized research on factual robustness, evaluation methodology, and defensive model development. Run experiments only against models and endpoints you own or are explicitly permitted to test.

## Out-of-Scope Use

Do not use this repository to:

- manipulate people or automate misinformation campaigns;
- contaminate public knowledge bases, search systems, or model feedback channels;
- evade safeguards on third-party services without authorization;
- target high-impact domains such as medicine, finance, elections, or public safety;
- publish credentials, private data, or unrestricted high-volume attack traces.

## Release Principles

The repository includes aggregate results and two minimized examples required to audit the report. Full experiment traces are excluded by default. Before releasing additional examples, verify that they are necessary for scientific scrutiny, remove credentials and service metadata, minimize optimized prompt content, and document why the benefit outweighs the misuse risk.

## Interpretation

A successful counterfactual output is evidence about behavior under the specified prompt and scoring procedure. It does not establish a persistent internal belief, general model unreliability, or real-world harm. Report model identifiers, dates, sample sizes, uncertainty, judge details, and known confounds alongside results.

## Provider Terms

Users are responsible for complying with model-provider terms, rate limits, applicable law, institutional policies, and research ethics requirements. The MIT software license does not grant permission to use third-party models or datasets outside their own terms.
