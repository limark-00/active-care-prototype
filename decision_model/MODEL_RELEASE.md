# Active Care V3.1 model asset

Release tag: `model-v3.1`

Asset: `active-care-v3.1-delta.zip`

The archive contains the partial-finetuning run directory used by the local text-decision service. It includes the 55 MB `best_delta.pt` checkpoint, run metadata, and evaluation reports. It does not contain patient data or the full MacBERT base model.

## Integrity

```text
Archive SHA256
f04357af3e131053974ddea4ce6dbc6daba2e25cf01304f8b772f59ef771723f

best_delta.pt SHA256
1bfb03f22cb5eb4840db8c33583391cf8209ecd544cae128de9748fcb8a0645e
```

## Install

Use the repository downloader rather than extracting the archive manually. It verifies both hashes and also prepares the pinned MacBERT base revision.

macOS:

```sh
bash 下载模型.command
```

Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\下载模型-Windows.ps1
```

The delta checkpoint requires `hfl/chinese-macbert-base` at commit `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`. Model outputs are for an industrial-design software prototype trained on synthetic labels. They are not clinically validated safety or care decisions.
