"""Generate GitHub-ready figures from a V6 paper run."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def generate_figures(run_dir: str | Path):
    run_dir = Path(run_dir); out = run_dir / "figures"; out.mkdir(parents=True, exist_ok=True)
    equity_path, trades_path, latency_path, signals_path = [run_dir/x for x in ("equity.csv","trades.csv","latency_stress.csv","signals.csv")]

    if equity_path.exists():
        d=pd.read_csv(equity_path)
        if not d.empty:
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(d.elapsed_minutes,d.equity); ax.set(title="Paper Equity During V6 Run",xlabel="Elapsed minutes",ylabel="Paper equity ($)"); ax.grid(alpha=.25); _save(fig,out/"01_equity_curve.png")
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(d.elapsed_minutes,d.capital_utilization*100); ax.set(title="Capital Utilization",xlabel="Elapsed minutes",ylabel="Locked capital (%)"); ax.grid(alpha=.25); _save(fig,out/"02_capital_utilization.png")
    if trades_path.exists():
        d=pd.read_csv(trades_path)
        booked=d[d.status.isin(["SIMULATED_FILL","PARTIAL_HEDGE","SIMULATED_LOSS","EXCESSIVE_HEDGE_MOVE"])] if not d.empty else d
        if not booked.empty:
            fig,ax=plt.subplots(figsize=(9,4.8)); labels=booked.subject.astype(str).str.slice(0,22); ax.bar(range(len(booked)),booked.conservative_pnl); ax.set_xticks(range(len(booked))); ax.set_xticklabels(labels,rotation=55,ha="right"); ax.set(title="Conservative Simulated P&L by Trade",ylabel="P&L ($)"); ax.axhline(0,linewidth=.8); _save(fig,out/"03_trade_pnl.png")
            topic=booked.groupby("topic",dropna=False).conservative_pnl.sum().sort_values(ascending=False)
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(topic.index.astype(str),topic.values); ax.set(title="Simulated P&L by Market Family",ylabel="P&L ($)"); ax.tick_params(axis="x",rotation=45); _save(fig,out/"04_pnl_by_category.png")
    if latency_path.exists():
        d=pd.read_csv(latency_path)
        if not d.empty:
            g=d.groupby("latency_seconds").agg(avg_net=("net_profit","mean"),positive_rate=("net_profit",lambda x:(x>0).mean())).reset_index()
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(g.latency_seconds,g.avg_net,marker="o"); ax.axhline(0,linewidth=.8); ax.set(title="Latency Stress Test",xlabel="Simulated inter-venue latency (s)",ylabel="Average conservative P&L ($)"); ax.grid(alpha=.25); _save(fig,out/"05_latency_stress.png")
            g.to_csv(out/"latency_stress_summary.csv",index=False)
    if signals_path.exists():
        d=pd.read_csv(signals_path)
        if not d.empty:
            r=d[~d.eligible].reason.value_counts().head(10).sort_values()
            if not r.empty:
                fig,ax=plt.subplots(figsize=(8,4.8)); ax.barh(r.index.astype(str),r.values); ax.set(title="Why Candidate Trades Were Rejected",xlabel="Count"); _save(fig,out/"06_rejection_reasons.png")
    print(f"GitHub-ready figures written to: {out}")
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run_dir"); a=ap.parse_args(); generate_figures(a.run_dir)
if __name__=="__main__": main()
