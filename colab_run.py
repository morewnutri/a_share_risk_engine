"""
A股风险评分引擎 - Google Colab 运行脚本
============================================
用法：将本文件内容粘贴到 Colab 单元格中运行，或直接上传后执行。

输出文件位于 /content/a_share_risk_engine/output/
  - latest_score.json       最新评分 JSON
  - factor_report.csv       因子明细 CSV
  - run_history.csv         历史评分（每次运行追加）
  - dashboard_report.html   自包含 HTML 仪表盘（可直接下载查看，无需 ngrok）
  - feature_snapshot.csv    特征快照

Colab 查看方式（无需 ngrok、无需账号）：
  运行结束后，左侧文件面板找到
    /content/a_share_risk_engine/output/dashboard_report.html
  右键 -> Download，用浏览器本地打开即可查看完整仪表盘。
"""

import os
import subprocess
import sys
from pathlib import Path

# ─── 配置 ────────────────────────────────────────────────────────────────────
REPO_URL   = "https://github.com/morewnutri/a_share_risk_engine.git"
REPO_DIR   = "/content/a_share_risk_engine"
BRANCH     = "copilot/fix-data-and-dashboard-issues"

# 可选：填写你的 FRED API Key 以获取美国实际利率等官方宏观数据
# 不填也能运行，但相关因子会标记为缺失并降低置信度
FRED_API_KEY = ""
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd, cwd=None, check=True):
    print(f"\n>>> {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, text=True, capture_output=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result


def show_text_file(path, max_chars=20000):
    path = Path(path)
    if path.exists():
        print(f"\n{'='*60}\n{path.name}\n{'='*60}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        print(text[:max_chars])
        if len(text) > max_chars:
            print("\n...[truncated]...")
    else:
        print(f"\nMissing: {path}")


# ── System deps ───────────────────────────────────────────────────────────────
run("apt-get update -y -q")
run("apt-get install -y -q graphviz")

# ── Clone repo ────────────────────────────────────────────────────────────────
if Path(REPO_DIR).exists():
    run(f"rm -rf {REPO_DIR}")
run(f"git clone --depth=1 {REPO_URL} {REPO_DIR}")
run(f"git checkout {BRANCH}", cwd=REPO_DIR)

# ── Install Python deps ───────────────────────────────────────────────────────
run(f"{sys.executable} -m pip install --upgrade pip -q", cwd=REPO_DIR)
run(f"{sys.executable} -m pip install -r requirements.txt -q", cwd=REPO_DIR)

# ── Set env ───────────────────────────────────────────────────────────────────
if FRED_API_KEY.strip():
    os.environ["FRED_API_KEY"] = FRED_API_KEY
    print("FRED_API_KEY set.")
else:
    print("FRED_API_KEY not set – FRED series (US10Y_REAL etc.) will be absent; "
          "engine will lower confidence accordingly.")

# ── Run engine ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running A-share risk engine ...")
print("="*60)
run(f"{sys.executable} a_share_risk_engine.py", cwd=REPO_DIR)

# ── Show outputs ──────────────────────────────────────────────────────────────
output_dir = Path(REPO_DIR) / "output"
state_dir  = Path(REPO_DIR) / "state"

print("\n=== Output files ===")
for p in sorted(output_dir.glob("*")):
    print(p)

print("\n=== State files ===")
for p in sorted(state_dir.glob("*")):
    print(p)

show_text_file(output_dir / "latest_score.json")
show_text_file(output_dir / "factor_report.csv")
show_text_file(output_dir / "run_history.csv")
show_text_file(state_dir  / "a_market_snapshot.csv")

# ── Colab download hint ───────────────────────────────────────────────────────
html_report = output_dir / "dashboard_report.html"
if html_report.exists():
    print("\n" + "="*60)
    print("✅ HTML dashboard ready – download and open in your browser:")
    print(f"   {html_report}")
    print()
    print("In Colab: left-side file panel -> navigate to the path above,")
    print("right-click the file -> Download.")
    print("No ngrok, no tunnel account required.")
    print("="*60)

    # Optionally show an inline Colab download link via IPython
    try:
        from google.colab import files as _colab_files
        print("\nColab Files API detected – attempting inline download trigger ...")
        _colab_files.download(str(html_report))
    except Exception:
        pass  # Not in Colab or files API unavailable – that's fine
