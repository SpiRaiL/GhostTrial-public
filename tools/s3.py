#!/usr/bin/env python3
"""Move files to and from Nebius Object Storage.

    .venv/bin/python tools/s3.py put <local> <key>
    .venv/bin/python tools/s3.py get <key> <local>
    .venv/bin/python tools/s3.py ls [prefix]

Nebius Object Storage is S3-compatible, so boto3 talks to it directly; there is no
`aws` CLI on this box and installing one is not worth it for three operations.

Credentials come from the environment (NB_KEY_ID / NB_KEY_SECRET) so they never
land in the repo. Retrieve them with:

    nebius iam v2 access-key get-secret --id <accesskey-...>

The upload here matters because the link out of this house is ~8.7 Mbps: pushing
the 40 GB training image was never an option, which is why the Nebius job builds
its environment from public sources (NGC base image, the GR00T repo on GitHub) and
only the few hundred MB that cannot be fetched publicly travel from here.
"""

import os
import sys
import threading

import boto3

ENDPOINT = "https://storage.eu-north1.nebius.cloud:443"
BUCKET = os.environ.get("NB_BUCKET", "rc-ghosttrial")

key, secret = os.environ.get("NB_KEY_ID"), os.environ.get("NB_KEY_SECRET")
if not key or not secret:
    raise SystemExit("set NB_KEY_ID and NB_KEY_SECRET "
                     "(nebius iam v2 access-key get-secret --id <id>)")

s3 = boto3.client("s3", endpoint_url=ENDPOINT, region_name="eu-north1",
                  aws_access_key_id=key, aws_secret_access_key=secret)


class Progress:
    """Bytes moved, printed on one line — uploads here take minutes, not seconds."""

    def __init__(self, total, label):
        self.total, self.seen, self.label = total, 0, label
        self.lock = threading.Lock()

    def __call__(self, n):
        with self.lock:
            self.seen += n
            pct = 100.0 * self.seen / self.total if self.total else 0
            sys.stdout.write(f"\r  {self.label}  {self.seen / 1e6:8.1f} / "
                             f"{self.total / 1e6:.1f} MB  {pct:5.1f}%")
            sys.stdout.flush()


cmd = sys.argv[1] if len(sys.argv) > 1 else "ls"

if cmd == "put":
    local, k = sys.argv[2], sys.argv[3]
    size = os.path.getsize(local)
    s3.upload_file(local, BUCKET, k, Callback=Progress(size, "up"))
    print(f"\n  -> s3://{BUCKET}/{k}")
elif cmd == "get":
    k, local = sys.argv[2], sys.argv[3]
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    size = s3.head_object(Bucket=BUCKET, Key=k)["ContentLength"]
    s3.download_file(BUCKET, k, local, Callback=Progress(size, "down"))
    print(f"\n  -> {local}")
elif cmd == "ls":
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    for o in r.get("Contents", []):
        print(f"  {o['Size'] / 1e6:9.2f} MB  {o['Key']}")
    if not r.get("Contents"):
        print(f"  (nothing under {prefix!r})")
else:
    raise SystemExit(__doc__)
