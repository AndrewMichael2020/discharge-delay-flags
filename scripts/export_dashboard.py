#!/usr/bin/env python3
"""Export or screenshot an Operational Delay Sentinel dashboard.

Always copies the source dashboard HTML to an export folder. If Playwright is
installed, it can also render a PNG screenshot without changing app code.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export dashboard HTML and optionally render a PNG screenshot.")
    p.add_argument("--dashboard", type=Path, default=Path("outputs/synthetic_90d_v1/operational_delay_dashboard.html"))
    p.add_argument("--out", type=Path, default=Path("exports"))
    p.add_argument("--png", action="store_true", help="Render PNG via Playwright if installed.")
    p.add_argument("--width", type=int, default=1440)
    p.add_argument("--height", type=int, default=1400)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dashboard = args.dashboard if args.dashboard.is_absolute() else root / args.dashboard
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    html_out = out / f"dashboard-export-{stamp}.html"
    shutil.copy2(dashboard, html_out)
    print(f"HTML export: {html_out}")
    if args.png:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            print("PNG screenshot skipped: Playwright is not installed.")
            print("Install later with: pip install playwright && python -m playwright install chromium")
            print(f"Import error: {exc}")
            return
        png_out = out / f"dashboard-screenshot-{stamp}.png"
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
            page.goto(html_out.resolve().as_uri(), wait_until="networkidle")
            page.screenshot(path=str(png_out), full_page=True)
            browser.close()
        print(f"PNG screenshot: {png_out}")


if __name__ == "__main__":
    main()
