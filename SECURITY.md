# Security Policy

## Supported Version

Security fixes apply to the current `main` branch. This research repository does not maintain multiple supported release lines.

## Reporting a Vulnerability

Do not open a public issue for exposed credentials, private endpoints, or a vulnerability that could materially increase misuse. Instead, use GitHub's private vulnerability reporting feature for this repository or contact the maintainer at `zcaiat@connect.ust.hk` with the subject `SAST-IR security report`.

Include the affected revision, reproduction conditions, impact, and a minimal proof of concept. Remove all live credentials and sensitive model outputs before sending the report.

## Credential Exposure

If an API key is committed or included in an artifact, revoke it at the provider immediately, remove it from active branches, and assess whether repository history must be rewritten. Deleting the visible file alone does not invalidate the credential.

## Research Misuse

Reports about prompt strategies that substantially increase real-world manipulation capability should follow the same private channel. See `RESPONSIBLE_USE.md` for the project's disclosure principles.
