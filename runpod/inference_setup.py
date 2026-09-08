"""Fetch and verify existing competition assets. Does not train or submit."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import requests

ROOT = Path("/workspace")


def download(kind, destination):
    if destination.exists():
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is None:
                print(kind, "already verified", destination.stat().st_size, flush=True)
                return
    urls = json.loads((ROOT / "download_manifest.json").read_text())
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(3):
        try:
            with requests.get(urls[kind], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                if expected and temporary.stat().st_size != int(expected):
                    raise ValueError("incomplete transfer")
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise ValueError("archive integrity check failed")
            temporary.replace(destination)
            print(kind, "verified", destination.stat().st_size, flush=True)
            return
        except Exception as error:
            # Signed URLs are access tokens: do not print request exceptions.
            print(kind, "attempt", attempt + 1, type(error).__name__, flush=True)
    raise RuntimeError(f"{kind} download failed integrity checks")


def main():
    ROOT.joinpath("assets").mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(download, "competition", ROOT / "assets/competition.zip"),
                pool.submit(download, "checkpoint", ROOT / "assets/exp002_best.pt"),
                pool.submit(subprocess.check_call, [sys.executable, "-m", "pip", "install", "-q",
                            "ultralytics==8.4.0", "pycocotools", "pandas", "pytest"])]
        for job in jobs:
            job.result()
    data = ROOT / "assets/competition"
    data.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT / "assets/competition.zip") as archive:
        for info in archive.infolist():
            if not (data / info.filename).resolve().is_relative_to(data.resolve()):
                raise ValueError("unsafe archive member")
        archive.extractall(data)
    import torch
    from ultralytics import YOLO
    checkpoint = ROOT / "assets/exp002_best.pt"
    model = YOLO(checkpoint)
    model.model.eval()
    model.model.requires_grad_(False)
    report = dict(
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_bytes=checkpoint.stat().st_size,
        parameters=sum(p.numel() for p in model.model.parameters()),
        model_args=model.model.args,
        cuda_available=torch.cuda.is_available(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable")
    # Force execution, rather than relying on CUDA availability alone.
    torch.ones(1, device="cuda").sum().item()
    ROOT.joinpath("assets/verified.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
