# Typed decisions on Chalk

Describe a situation once, as text or a JSON record, and ask a decision model several questions
about it. Each question has a type that fixes the shape of its answer, and every question is
answered from the same forward pass on one GPU, deployed with
[`chalkcompute`](https://pypi.org/project/chalkcompute/).

The model is [`laya-typed-decisions`](https://huggingface.co/michaelfeil/laya-typed-decisions), a
421M-parameter ModernBERT-large decision model fine-tuned on four workflows: customer support,
invoice processing, security alerts and agent-trace observability.

| Type | You write | You get back |
| --- | --- | --- |
| `choice` | `criteria`: `{option: description}` | `choice` and a probability per option |
| `noul` | a statement as `instructions` | `noul`: the probability that the statement is true |
| `score` | `criteria`: an ordered list of levels | a distribution over levels and its expected `score` |

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Builds the image (torch, laya, weights baked in) and deploys a `chalkcompute.ScalingGroup` on one GPU. |
| `server.py` | FastAPI app: `POST /decide`, `GET /status`, and a small playground at `GET /`. |
| `index.html` | The playground page. |
| `example.py` | Calls `/decide` with three typed questions about a support ticket. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule GPU workloads.
Credentials come from the SDK’s ambient credentials from `chalk login`. You can also set `CHALK_API_SERVER`,
`CHALK_CLIENT_ID`, `CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID` yourself.

```bash
./deploy.py
./example.py https://<web_url>
```

Or with curl:

```bash
curl -s https://<web_url>/decide -H 'content-type: application/json' -d '{
  "state": {"ticket": "Charged twice for our annual renewal. Refund it today or we leave."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, invoices and refunds",
                          "technical": "The product is not working",
                          "account": "Login, plan and access"}},
    "needs_human": {"type": "noul",
                    "instructions": "A human agent should take over this conversation."},
    "churn_risk": {"type": "score", "instructions": "How likely is this customer to leave?",
                   "criteria": ["low", "moderate", "high", "imminent"]}
  }
}'
```

That request answers `billing` (p = 0.86), `needs_human` 0.68 and `churn_risk` 2.48 out of 3.

## Measured

- **Accuracy:** on the 2,000 decisions in the
  [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test split, the
  model gives the reference answer 76.6% of the time. By type, that is 85.7% for `noul`, 73.3% for
  `choice` and 72.3% for `score`. Always picking each question's most common training answer
  scores 48.3%.
- **Latency on one NVIDIA L4:** 24 ms of model time for one question and 46 ms for five about the
  same security alert. The record is encoded once, so extra questions add little.

## Notes

- **It is a specialist.** Expect weaker answers outside the four workflows it was tuned on. Write
  states and questions in the shape of its training data, which uses JSON records and `noul`
  questions phrased as statements.
- **Treat the probabilities as indicative.** The checkpoint's calibration temperatures were fitted
  on training data, and the library warns about one of them at load time. Refit on your own
  held-out decisions before you threshold on them.
- **The image build uses `uv` and pins torch.** With plain `pip` and an unpinned torch, the CUDA
  image took too long to build.
- **The demo endpoint is public.** `/decide` caps input size, question count and options, and rate
  limits each client to 60 requests a minute (`DECISIONS_RATE_LIMIT_PER_MINUTE`). Browsers may call
  it only from origins matching `DECISIONS_ALLOWED_ORIGIN_REGEX`. Delete the group when you are done:

  ```bash
  uv run --with chalkcompute python -c \
    "from chalkcompute import ScalingGroup; ScalingGroup.from_name('typed-decisions').delete()"
  ```

## License

The code is Apache-2.0. The `laya-typed-decisions` weights and the typed-decisions dataset are
Apache-2.0.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
