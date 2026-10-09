#!/usr/bin/env python3
import json
import os

import boto3


def main() -> None:
    if os.environ.get("WANDB_API_KEY"):
        print("[wandb_setup] WANDB_API_KEY already set")
        return
    if os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled"}:
        print(f"[wandb_setup] WANDB_MODE={os.environ.get('WANDB_MODE')}; not loading API key")
        return

    secret_id = os.environ.get("WANDB_SECRET_ID", "/core-cn/model_trainer/team-wandb-password")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    try:
        client = boto3.client("secretsmanager", region_name=region)
        raw = client.get_secret_value(SecretId=secret_id)["SecretString"]
        value = json.loads(raw)
        api_key = value.get("WANDB_API_KEY")
        if not api_key:
            raise ValueError(f"secret {secret_id} does not contain WANDB_API_KEY")
    except Exception as exc:
        print(f"[wandb_setup] Could not load W&B API key: {exc}")
        print("[wandb_setup] Disabling W&B so training can continue")
        print("export WANDB_MODE=disabled")
        return

    print("export WANDB_API_KEY=" + json.dumps(api_key))
    print("[wandb_setup] W&B API key loaded", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
