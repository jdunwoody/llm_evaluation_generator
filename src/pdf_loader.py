from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz


def parse_page(page) -> str:
    lines = []

    for b in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
        for l in b["lines"]:
            spans = []

            for s in l["spans"]:
                text = s["text"].strip()
                if len(text) == 0:
                    continue

                spans += [text]

            if len(spans) == 0:
                continue

            line = " ".join(spans)

            lines.append(line)

        if len(lines) == 0:
            continue

    return "\n".join(lines)


def to_markdown(input_file):
    doc = fitz.open(input_file)

    pages = [parse_page(page) for page in doc.pages()]

    return "\n\n".join(pages)


def _main():
    base_dir = Path(__file__).parents[2]
    data_dir = base_dir / "data"
    input_file = data_dir / "JPM Electravision 14th Annual Energy Paper 20240305.pdf"

    output = to_markdown(input_file=input_file)

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.name}.txt"
    output_file.write_text(output)


if __name__ == "__main__":
    _main()
