#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Ask the deployed server three typed questions about a support ticket.

./example.py https://<web_url>
"""

from __future__ import annotations

import json
import sys

import httpx

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, invoices and refunds",
            "technical": "The product is not working",
            "account": "Login, plan and access",
        },
    },
    # A noul question is a statement; the answer is the probability that it is true.
    "needs_human": {
        "type": "noul",
        "instructions": "A human agent should take over this conversation.",
    },
    "churn_risk": {
        "type": "score",
        "instructions": "How likely is this customer to leave?",
        "criteria": ["low", "moderate", "high", "imminent"],
    },
}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    response = httpx.post(
        f"{sys.argv[1].rstrip('/')}/decide",
        json={
            "state": {
                "ticket": "Charged twice for our annual renewal. Refund it today or we leave.",
            },
            "questions": QUESTIONS,
        },
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    answers = body["answers"]
    print("team:       ", answers["team"]["choice"], answers["team"]["probabilities"])
    print("needs_human:", answers["needs_human"]["noul"])
    print("churn_risk: ", answers["churn_risk"]["score"], answers["churn_risk"]["legend"])
    print(f"model time:  {body['infer_ms']} ms")
    print(json.dumps(body["usage"]))


if __name__ == "__main__":
    main()
