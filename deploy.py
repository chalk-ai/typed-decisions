#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.1"]
# ///
"""Deploy the typed-decisions server (server.py + index.html) as a GPU ScalingGroup.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py
Then POST to <web_url>/decide, or open <web_url> for a small playground.
"""

from __future__ import annotations

import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "michaelfeil/laya-typed-decisions"
# Pinned so a rebuilt image serves the same weights the demo was checked against.
MODEL_REVISION = "2741340228fc0c982ae9dd63b34dcd5a9e7ce3ce"
HF_HOME = "/models/hf"


def main() -> None:
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
        name='typed-decisions',
        gpu='nvidia-l4',
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
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=2400.0, ready_timeout=1800.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
