"""Generate GitHub-ready research figures from a V10.0 Final paper run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_figures(run_dir: str | Path):
    run_dir = Path(run_dir)
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    ep = run_dir / "equity.csv"
    tp = run_dir / "trades.csv"
    lp = run_dir / "latency_stress.csv"
    sp = run_dir / "signals.csv"
    up = run_dir / "early_unwinds.csv"
    cp = run_dir / "capacity_analysis.csv"
    pp = run_dir / "allocator_policy_analysis.csv"
    dp = run_dir / "duration_buckets.csv"

    if ep.exists():
        d = pd.read_csv(ep)
        if not d.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(d.elapsed_minutes, d.equity)
            ax.set(title="V10.0 Final Paper Equity", xlabel="Elapsed minutes", ylabel="Paper equity ($)")
            ax.grid(alpha=.25)
            _save(fig, out / "01_equity_curve.png")

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(d.elapsed_minutes, d.capital_utilization * 100)
            ax.set(title="Capital Utilization", xlabel="Elapsed minutes", ylabel="Locked capital (%)")
            ax.grid(alpha=.25)
            _save(fig, out / "02_capital_utilization.png")

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(d.elapsed_minutes, d.locked_profit, label="Locked-to-resolution profit")
            ax.plot(d.elapsed_minutes, d.realized_pnl, label="Realized exit/settlement P&L")
            ax.set(title="Locked vs Realized Paper Profit", xlabel="Elapsed minutes", ylabel="P&L ($)")
            ax.legend()
            ax.grid(alpha=.25)
            _save(fig, out / "03_locked_vs_realized_profit.png")

            bucket_cols = {
                "0-30d": "bucket_0_30d",
                "31-90d": "bucket_31_90d",
                "91-365d": "bucket_91_365d",
                ">365d": "bucket_gt365d",
            }
            available = {k: v for k, v in bucket_cols.items() if v in d.columns}
            if available:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                for label, col in available.items():
                    ax.plot(d.elapsed_minutes, d[col], label=label)
                ax.set(title="Locked Capital by Resolution Horizon", xlabel="Elapsed minutes", ylabel="Capital ($)")
                ax.legend(title="Settlement bucket")
                ax.grid(alpha=.25)
                _save(fig, out / "04_duration_bucket_capital.png")

    if tp.exists():
        d = pd.read_csv(tp)
        if not d.empty:
            labels = d.subject.astype(str).str.slice(0, 22)
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.bar(range(len(d)), d.hold_profit)
            ax.set_xticks(range(len(d)))
            ax.set_xticklabels(labels, rotation=55, ha="right")
            ax.set(title="Locked Profit by Booked Position", ylabel="Profit at resolution ($)")
            ax.axhline(0, linewidth=.8)
            _save(fig, out / "05_position_profit.png")

            g = d.groupby("topic", dropna=False).hold_profit.sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(g.index.astype(str), g.values)
            ax.set(title="Locked Profit by Market Family", ylabel="Profit ($)")
            ax.tick_params(axis="x", rotation=45)
            _save(fig, out / "06_profit_by_category.png")

            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.scatter(d.settlement_days, d.annualized_hold_return * 100, s=45)
            ax.set(title="Capital Lock vs Annualized Hold Return", xlabel="Estimated days until resolution", ylabel="Simple annualized return (%)")
            ax.grid(alpha=.25)
            _save(fig, out / "07_lock_horizon_vs_apr.png")

            if "capital_velocity" in d.columns:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                ax.scatter(d.settlement_days, d.capital_velocity, s=45)
                ax.set(title="Capital Velocity vs Lock Horizon", xlabel="Estimated days until resolution", ylabel="Capital-velocity score")
                ax.grid(alpha=.25)
                _save(fig, out / "08_capital_velocity.png")

            if "duration_bucket" in d.columns:
                g = d.groupby("duration_bucket").locked_capital.sum().reindex(["0-30d", "31-90d", "91-365d", ">365d"]).dropna()
                if not g.empty:
                    fig, ax = plt.subplots(figsize=(7.5, 4.5))
                    ax.bar(g.index.astype(str), g.values)
                    ax.set(title="Booked Capital by Resolution Horizon", xlabel="Resolution horizon", ylabel="Locked capital ($)")
                    _save(fig, out / "09_booked_capital_by_horizon.png")

            # Presentation benchmark comparison. 5.5% is a pragmatic target
            # assumption, not a forecast of either strategy or future S&P 500 returns.
            if "annualized_hold_return" in d.columns and "locked_capital" in d.columns:
                valid = d[(d.locked_capital > 0) & d.annualized_hold_return.notna()]
                if not valid.empty:
                    weighted_apr = float((valid.annualized_hold_return * valid.locked_capital).sum() / valid.locked_capital.sum()) * 100
                    benchmark = 5.5
                    fig, ax = plt.subplots(figsize=(7.0, 4.5))
                    ax.bar(["V13 paper hold APR", "5.5% presentation target"], [weighted_apr, benchmark])
                    ax.set(title="Annualized Hold Return vs Research Benchmark", ylabel="Annualized return (%)")
                    _save(fig, out / "09b_apr_vs_benchmark.png")

            if "equivalence_score" in d.columns:
                q = pd.to_numeric(d.equivalence_score, errors="coerce").dropna()
                if not q.empty:
                    fig, ax = plt.subplots(figsize=(7.5, 4.5))
                    ax.hist(q, bins=min(10, max(3, len(q))))
                    ax.set(title="Booked Trade Equivalence Scores", xlabel="Strict semantic equivalence score", ylabel="Booked positions")
                    _save(fig, out / "09c_equivalence_scores.png")

            if "match_source" in d.columns:
                m = d.match_source.fillna("unknown").value_counts().sort_values()
                if not m.empty:
                    fig, ax = plt.subplots(figsize=(7.5, 4.5))
                    ax.barh(m.index.astype(str), m.values)
                    ax.set(title="Booked Trades by Equivalence Path", xlabel="Booked positions")
                    _save(fig, out / "09d_match_source.png")

    if lp.exists():
        d = pd.read_csv(lp)
        if not d.empty:
            g = d.groupby("latency_seconds").agg(
                avg_net=("net_profit", "mean"),
                positive_rate=("net_profit", lambda x: (x > 0).mean()),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(g.latency_seconds, g.avg_net, marker="o")
            ax.axhline(0, linewidth=.8)
            ax.set(title="Latency Stress Test", xlabel="Inter-venue latency (s)", ylabel="Average conservative P&L ($)")
            ax.grid(alpha=.25)
            _save(fig, out / "10_latency_stress.png")
            g.to_csv(out / "latency_stress_summary.csv", index=False)

    if sp.exists():
        d = pd.read_csv(sp)
        if not d.empty:
            r = d[~d.eligible].reason.value_counts().head(12).sort_values()
            if not r.empty:
                fig, ax = plt.subplots(figsize=(8.5, 5.2))
                ax.barh(r.index.astype(str), r.values)
                ax.set(title="Why Candidate Trades Were Rejected", xlabel="Count")
                _save(fig, out / "11_rejection_reasons.png")

            eligible = d[d.eligible == True].copy()
            if not eligible.empty and "capital_velocity" in eligible.columns:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                ax.scatter(eligible.settlement_days, eligible.capital_velocity, s=25, alpha=.6)
                ax.set(title="Eligible Signals: Duration vs Capital Velocity", xlabel="Days until resolution", ylabel="Capital velocity")
                ax.grid(alpha=.25)
                _save(fig, out / "12_eligible_signal_velocity.png")

    if up.exists():
        d = pd.read_csv(up)
        if not d.empty:
            counts = d.action.value_counts()
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.bar(counts.index.astype(str), counts.values)
            ax.set(title="Early-Unwind Decisions", ylabel="Checks")
            _save(fig, out / "13_early_unwind_decisions.png")

    if cp.exists():
        d = pd.read_csv(cp)
        if not d.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(d.bankroll, d.estimated_locked_profit, marker="o")
            ax.set(title="Observed-Book Capacity Estimate", xlabel="Counterfactual bankroll ($)", ylabel="Estimated locked profit ($)")
            ax.grid(alpha=.25)
            _save(fig, out / "14_capacity_analysis.png")

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(d.bankroll, d.weighted_settlement_days, marker="o")
            ax.set(title="Capacity vs Capital Lock Horizon", xlabel="Counterfactual bankroll ($)", ylabel="Capital-weighted settlement days")
            ax.grid(alpha=.25)
            _save(fig, out / "15_capacity_lock_horizon.png")

    if pp.exists():
        d = pd.read_csv(pp)
        if not d.empty:
            order = ["raw_profit", "roc", "capital_velocity", "duration_biased"]
            d["policy"] = pd.Categorical(d.policy, categories=order, ordered=True)
            d = d.sort_values("policy")
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.bar(d.policy.astype(str), d.estimated_locked_profit)
            ax.set(title="Allocator Policy Comparison", xlabel="Allocation policy", ylabel="Estimated locked profit ($)")
            ax.tick_params(axis="x", rotation=20)
            _save(fig, out / "16_allocator_policy_profit.png")

            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.bar(d.policy.astype(str), d.weighted_settlement_days)
            ax.set(title="Allocator Policy Lock-Time Comparison", xlabel="Allocation policy", ylabel="Capital-weighted settlement days")
            ax.tick_params(axis="x", rotation=20)
            _save(fig, out / "17_allocator_policy_lock_time.png")

            d.to_csv(out / "allocator_policy_summary.csv", index=False)

    if dp.exists():
        d = pd.read_csv(dp)
        if not d.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.bar(d.duration_bucket.astype(str), d.eligible_unique_signals)
            ax.set(title="Eligible Opportunities by Resolution Horizon", xlabel="Resolution horizon", ylabel="Unique eligible signals")
            _save(fig, out / "18_eligible_by_duration.png")

    print(f"GitHub-ready V10 figures written to: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    generate_figures(a.run_dir)


if __name__ == "__main__":
    main()
