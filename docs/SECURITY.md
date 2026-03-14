# Security — DSP501

> Version: 1.0 | Last updated: 2026-03-13

## Threat Model

This is an academic/research project with minimal attack surface.

| Risk | Level | Mitigation |
|------|-------|------------|
| Data leakage | Low | UrbanSound8K is public dataset |
| Secrets exposure | Low | No API keys required; .env.example provided |
| Supply chain | Medium | Pin dependency versions in requirements.txt |
| Model poisoning | Low | Dataset is fixed; no user-uploaded data |

## Privacy

- No user data collected
- No tracking or analytics
- UrbanSound8K licensed under CC BY-NC 3.0
- All processing is local

## Secrets Management

- `.env` file excluded via `.gitignore`
- `.env.example` provided as template
- No API keys or credentials needed

## Related Docs

- [DEPLOYMENT.md](DEPLOYMENT.md)
