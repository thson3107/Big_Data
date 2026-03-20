from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def export_summary_report(report_path: str, payload: dict):
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def markdown_summary(path: str, sections: dict):
    lines = ["# Bao cao tong hop", ""]
    for title, text in sections.items():
        lines.append(f"## {title}")
        lines.append(str(text))
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def pdf_summary(path: str, sections: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "BAO CAO TONG HOP", fontsize=18, fontweight="bold", transform=ax.transAxes)
    y -= 0.05
    for title, text in sections.items():
        ax.text(0.02, y, str(title), fontsize=12, fontweight="bold", transform=ax.transAxes)
        y -= 0.03
        for line in str(text).split("\n"):
            ax.text(0.04, y, line[:120], fontsize=10, transform=ax.transAxes)
            y -= 0.024
            if y < 0.05:
                break
        y -= 0.015
        if y < 0.05:
            break
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
