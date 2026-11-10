import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def relative_files(paths):
    """Store repository-relative paths in the data manifest."""
    return [path.relative_to(ROOT).as_posix() for path in paths]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_records(path):
    """Read JSONL even when a source text field contains a raw newline."""
    rows = []
    buffer = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        buffer += line
        try:
            rows.append(json.loads(buffer))
            buffer = ""
        except json.JSONDecodeError:
            continue
    if buffer.strip():
        raise ValueError(f"Unparsed JSON content remains in {path}")
    return rows


def prepare_tnews(source=None):
    out = DATA / "tnews"
    out.mkdir(parents=True, exist_ok=True)
    files = [out / name for name in ["train.json", "dev.json", "labels.json"]]
    if all(path.exists() for path in files):
        source_name = "prepared copy in data/tnews"
    else:
        source = Path(source) if source else DATA / "source" / "tnews_public"
        if not (source / "train.json").exists():
            raise FileNotFoundError(
                "TNEWS source was not found. Pass --tnews-source with the "
                "official tnews_public directory."
            )
        source_name = "CLUE tnews_public"
        for name in ["train.json", "dev.json", "labels.json"]:
            shutil.copy2(source / name, out / name)
    return {
        "name": "TNEWS",
        "source": source_name,
        "source_url": "https://github.com/CLUEbenchmark/CLUE",
        "files": relative_files(files),
    }


def prepare_chnsenticorp():
    out = DATA / "chnsenticorp"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "train.jsonl").exists():
        ds = load_dataset("lansinuote/ChnSentiCorp")
        for split in ["train", "validation", "test"]:
            rows = [{"text": str(r["text"]), "label": int(r["label"])} for r in ds[split]]
            write_jsonl(out / f"{split}.jsonl", rows)
    return {
        "name": "ChnSentiCorp",
        "source": "lansinuote/ChnSentiCorp",
        "source_url": "https://huggingface.co/datasets/lansinuote/ChnSentiCorp",
        "files": relative_files([out / x for x in ["train.jsonl", "validation.jsonl", "test.jsonl"]]),
    }


def prepare_iflytek(source=None):
    out = DATA / "iflytek"
    out.mkdir(parents=True, exist_ok=True)
    files = [out / name for name in ["train.json", "dev.json", "test.json", "labels.json"]]
    if all(path.exists() for path in files):
        source_name = "prepared copy in data/iflytek"
    else:
        source = Path(source) if source else DATA / "source" / "iflytek_public"
        if not (source / "train.json").exists():
            raise FileNotFoundError(
                "IFLYTEK source was not found. Pass --iflytek-source to include it."
            )
        source_name = "CLUE iflytek_public"
        for name in ["train.json", "dev.json", "test.json", "labels.json"]:
            shutil.copy2(source / name, out / name)
    return {
        "name": "IFLYTEK",
        "source": source_name,
        "source_url": "https://github.com/CLUEbenchmark/CLUE",
        "files": relative_files(files),
        "reader": "robust_jsonl_raw_newline_join",
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare the benchmark data used by the experiments.")
    parser.add_argument("--tnews-source", type=Path, help="Directory containing the CLUE TNEWS files.")
    parser.add_argument("--iflytek-source", type=Path, help="Optional directory containing the CLUE IFLYTEK files.")
    parser.add_argument("--include-iflytek", action="store_true", help="Prepare IFLYTEK in addition to the paper datasets.")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    datasets = [prepare_tnews(args.tnews_source), prepare_chnsenticorp()]
    if args.include_iflytek or args.iflytek_source:
        datasets.append(prepare_iflytek(args.iflytek_source))
    manifest = {"datasets": datasets}
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
