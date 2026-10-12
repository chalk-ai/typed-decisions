#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.1"]
# ///
"""Deploy the typed-decisions server (server.py + index.html) as a GPU ScalingGroup.

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l4]
Then POST to <web_url>/decide, or open <web_url> for a small playground.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "michaelfeil/laya-typed-decisions"
# Pinned so a rebuilt image serves the same weights the demo was checked against.
MODEL_REVISION = "2741340228fc0c982ae9dd63b34dcd5a9e7ce3ce"
HF_HOME = "/models/hf"


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="typed-decisions")
    # Defaults to the environment in `chalk config` (or CHALK_ENVIRONMENT_ID).
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(["uv==0.11.8"])  # the slim base image does not ship uv
        # uv installs torch's CUDA wheels in parallel, which keeps the build well inside the
        # image builder's deadline; plain pip does not. torch is pinned for the same reason.
        .uv_pip_install(
            [
                "torch==2.7.1",
                "laya==0.4.6",
                "transformers==5.19.0",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
            ]
        )
        .env({"HF_HOME": HF_HOME, "USE_TF": "0"})
        # Baking the weights in keeps a cold replica off Hugging Face. The layer sits before
        # the app files, so code edits reuse it.
        .run_commands(
            f'HF_HOME={HF_HOME} python -c "from huggingface_hub import snapshot_download; '
            + f"snapshot_download('{MODEL_ID}', revision='{MODEL_REVISION}')\""
        )
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
        .workdir("/app")
    )

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        cpu="4",
        memory="16Gi",
        env={
            "DECISIONS_MODEL": MODEL_ID,
            "DECISIONS_MODEL_REVISION": MODEL_REVISION,
            "HF_HUB_OFFLINE": "1",
            "PORT": "8080",
        },
        entrypoint=["python", "/app/server.py"],
        port=8080,
        # Requests are stateless, so any replica can answer any request.
        min_replicas=args.replicas,
        max_replicas=args.replicas,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
