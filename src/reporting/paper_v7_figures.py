"""Generate GitHub-ready research figures from a V7 paper run."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def generate_figures(run_dir: str|Path):
    run_dir=Path(run_dir); out=run_dir/"figures"; out.mkdir(parents=True,exist_ok=True)
    ep,tp,lp,sp,up,cp=[run_dir/x for x in ("equity.csv","trades.csv","latency_stress.csv","signals.csv","early_unwinds.csv","capacity_analysis.csv")]
    if ep.exists():
        d=pd.read_csv(ep)
        if not d.empty:
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(d.elapsed_minutes,d.equity); ax.set(title="Settlement-Aware Paper Equity",xlabel="Elapsed minutes",ylabel="Paper equity ($)"); ax.grid(alpha=.25); _save(fig,out/"01_equity_curve.png")
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(d.elapsed_minutes,d.capital_utilization*100); ax.set(title="Capital Utilization",xlabel="Elapsed minutes",ylabel="Locked capital (%)"); ax.grid(alpha=.25); _save(fig,out/"02_capital_utilization.png")
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(d.elapsed_minutes,d.locked_profit,label="Locked-to-resolution profit"); ax.plot(d.elapsed_minutes,d.realized_pnl,label="Realized exit/settlement P&L"); ax.set(title="Locked vs Realized Paper Profit",xlabel="Elapsed minutes",ylabel="P&L ($)"); ax.legend(); ax.grid(alpha=.25); _save(fig,out/"03_locked_vs_realized_profit.png")
    if tp.exists():
        d=pd.read_csv(tp)
        if not d.empty:
            labels=d.subject.astype(str).str.slice(0,22); fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(range(len(d)),d.hold_profit); ax.set_xticks(range(len(d))); ax.set_xticklabels(labels,rotation=55,ha="right"); ax.set(title="Locked Profit by Booked Position",ylabel="Profit at resolution ($)"); ax.axhline(0,linewidth=.8); _save(fig,out/"04_position_profit.png")
            g=d.groupby("topic",dropna=False).hold_profit.sum().sort_values(ascending=False); fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(g.index.astype(str),g.values); ax.set(title="Locked Profit by Market Family",ylabel="Profit ($)"); ax.tick_params(axis="x",rotation=45); _save(fig,out/"05_profit_by_category.png")
            fig,ax=plt.subplots(figsize=(8,4.8)); ax.scatter(d.settlement_days,d.annualized_hold_return*100,s=40); ax.set(title="Capital Lock vs Annualized Hold Return",xlabel="Estimated days until resolution",ylabel="Simple annualized return (%)"); ax.grid(alpha=.25); _save(fig,out/"06_lock_horizon_vs_apr.png")
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.hist(d.settlement_days,bins=min(12,max(3,len(d)))); ax.set(title="Estimated Capital Lock Horizon",xlabel="Days until resolution",ylabel="Booked positions"); _save(fig,out/"07_settlement_horizon.png")
    if lp.exists():
        d=pd.read_csv(lp)
        if not d.empty:
            g=d.groupby("latency_seconds").agg(avg_net=("net_profit","mean"),positive_rate=("net_profit",lambda x:(x>0).mean())).reset_index(); fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(g.latency_seconds,g.avg_net,marker="o"); ax.axhline(0,linewidth=.8); ax.set(title="Latency Stress Test",xlabel="Inter-venue latency (s)",ylabel="Average conservative P&L ($)"); ax.grid(alpha=.25); _save(fig,out/"08_latency_stress.png"); g.to_csv(out/"latency_stress_summary.csv",index=False)
    if sp.exists():
        d=pd.read_csv(sp)
        if not d.empty:
            r=d[~d.eligible].reason.value_counts().head(10).sort_values();
            if not r.empty:
                fig,ax=plt.subplots(figsize=(8,4.8)); ax.barh(r.index.astype(str),r.values); ax.set(title="Why Candidate Trades Were Rejected",xlabel="Count"); _save(fig,out/"09_rejection_reasons.png")
    if up.exists():
        d=pd.read_csv(up)
        if not d.empty:
            counts=d.action.value_counts(); fig,ax=plt.subplots(figsize=(7,4.2)); ax.bar(counts.index.astype(str),counts.values); ax.set(title="Early-Unwind Decisions",ylabel="Checks"); _save(fig,out/"10_early_unwind_decisions.png")
    if cp.exists():
        d=pd.read_csv(cp)
        if not d.empty:
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(d.bankroll,d.estimated_locked_profit,marker="o"); ax.set(title="Observed-Book Capacity Estimate",xlabel="Counterfactual bankroll ($)",ylabel="Estimated locked profit ($)"); ax.grid(alpha=.25); _save(fig,out/"11_capacity_analysis.png")
    print(f"GitHub-ready V7 figures written to: {out}"); return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run_dir"); a=ap.parse_args(); generate_figures(a.run_dir)
if __name__=="__main__": main()
