# Cross-Market Prediction Arbitrage Engine
This is a precision-first engine for identifying cross-market arbitrage opportunities between Kalshi and Polymarket using semantic contract matching, settlement verification, live pricing, and execution-aware return filtering.

![Candidate filtering pipeline](figures/02_candidate_filter_sankey_clean.png)

## Overview

This project focuses on identifying economically equivalent contracts listed on different prediction markets and then monitoring them in real time to detect price changes that could create arbitrage opportunities with a guaranteed settlement profit if executed as intended.

The largest challenge in the project was certainly the contract matching, as a single false positive in pair evaluation meant a false arbitrage would likely undo any of the marginal profits made by genuine opportunities. Rigorous evaluation was required to ensure that matched contracts were equivalent not only in their titles, but also in their platform-specific clauses.

The system combines a trained semantic recognition model and layers deterministic precision filters for extremely high matching accuracy. These pairs are then checked for verification based on their settlement rules, with the passed pairs being checked for financial promise and watch-listed to be monitored over a longer cycle.

## Project Motivation

The idea originally began as a simple Kalshi-only arbitrage engine, and while genuine pricing errors could be detected, they were rare and offered extremely low returns after fees. The approach also depended heavily on temporary pricing discrepancies within a single venue, making it difficult to build into a reliable long-term strategy. This led me to use another major prediction market, Polymarket, as a source of cross-market opportunities.

In theory, if I can buy opposing outcomes across the two platforms for less than the guaranteed settlement payout, the difference represents an arbitrage opportunity. However two contracts can look almost identical while differing in settlement date, event definition, cancellation rules, eligible outcomes, geographical scope, competition stage, or resolution procedure. This meant the project focus shifted away from simply finding price discrepancies, and towards building a system that could determine if two contracts were genuinely compatible enough to trade as a hedge.

## System Architecture

Although many designs were tested, this was the final architecture chosen because it provided the strongest balance between high pairing precision and retaining a useful level of recall.

![System architecture](figures/01_system_architecture_final.png)

## Market Matching Pipeline

### 1. Candidate Retrieval with BGE Embeddings

The first big problem was scale. Comparing each Kalshi market against every active Polymarket market directly would have O(N²) time complexity, which would not be viable with thousands of markets on each side. To avoid this, I used BGE sentence embeddings as a semantic retrieval layer. This method converts sentences into high-dimensional vectors to group semantically similar sentences together. It is not accurate enough to determine exact matches on its own; however, it helps narrow the search space for more precise pair analysis.

In the final run, this retrieval stage produced over 35,000 candidate pairs, all to be passed to the next classifier.

### 2. Pairwise Equivalence Classification with DeBERTa

This stage was the main learned component of the system.

This more complex model would be useful for distinguishing between contracts that are topically similar and contracts which represent the same proposition. This is especially useful here, as with so many contracts, there are bound to be pairs which refer to the same person and result but differ in context which would be easily missed by many systems.

The final production classifier used Microsoft's DeBERTa V3 Base, giving these results on a validation set of 669 pairs.

| Metric | Validation Result |
|---|---:|
| Accuracy | **89.99%** |
| Precision | **100.00%** |
| Recall | **75.64%** |
| F1 Score | **86.13%** |
| PR-AUC | **99.89%** |
| ROC-AUC | **99.93%** |
| Validation Loss | **0.4157** |

*Validation set: n = 669. Precision was prioritised over recall because a false-positive equivalence match could create basis risk in a supposedly hedged trade.*

In the final experiment, it narrowed the 35,525 candidates down to 10,525 passes. This removed a large amount of obvious noise; however, there were still likely to be errors, so this result could not be used directly to make trades. The high validation score could not guarantee that each of the ten thousand passes had exactly compatible structure or settlement conditions.

![Retention by DeBERTa score decile](figures/09_retention_by_score_decile_journal.png)

### 3. Deterministic Semantic and Precision Filtering

After DeBERTa, I added deterministic filters focused on structural differences that the model could have missed. This final filtering stack consisted of V7, V8, and V8.1, increasing confidence that pairs making it through were genuine matches.

The V7 acts as a broad structural safeguard. It checks details such as event identity, subject identity, competition, stage, date, structural wording patterns and known mismatch types. This stage was not particularly restrictive, focusing on removing high-confidence contradictions and allowing all plausible candidates to continue.

The V7 reduced the 10,525 candidates to approximately 7,208.

V8, on the other hand, was much stricter. This stage checked many of the same features as V7 while also examining extra details such as geographical scope, metric, action, time periods, and granularity. It also had a third outcome beyond pass or reject: an "insufficient evidence" classification. This prevented forced positive decisions when one side of the pair was not as detailed as the other.

Of the now 7,208 candidates, 1,414 passed, 1,374 contained an explicit mismatch and 4,420 had insufficient evidence. This is where the pipeline became deliberately conservative to prevent any risk of mismatch.

The final V8.1 was a precision guard added after manually examining the types of false positives that slipped through the previous filters. This removed mismatches such as:
- "become PM" vs. "be the next PM"
- "next to leave" vs. "first to leave"
- actor-only vs. actor-and-programme award markets
- misleading keyword overlap between unrelated metrics

This removed a further 78 candidates, leaving 1,336 of the initial 35,000+ pairs remaining.

This process was deliberately biased towards precision rather than recall, as a false negative may simply mean missing a potential arbitrage opportunity; however, a single false positive could indicate a false arbitrage opportunity, leading to exposed risk and undermining the purpose of the project.

![Example semantic match and mismatch cases](figures/07_match_case_studies_compact.png)

I also examined how the filtering stages impacted different prediction-market domains.

![Domain pipeline survival heatmap](figures/08_domain_pipeline_survival_heatmap.png)

## Settlement and Execution Filtering

### Settlement-Rule Verification

Another unexpected finding was that even perfect semantic equivalence is not enough. Even if two contracts describe the exact same thing, they may have different rules governing how they resolve. For all candidates that survived semantic filtering and later displayed favourable live pricing, I retrieved the settlement rules from both Kalshi and Polymarket. This allowed me to check for differences including settlement deadlines, cancellation provisions, fallback resolution procedures, and event-completion requirements.

During the final run this brought the 46 price-confirmed opportunities down to 26 passing watch candidates.

### Live Pricing and Execution Checks

Only after the semantic and settlement stages could the system move into live execution analysis.

This required the system to evaluate:
- Executable prices
- Platform fees
- Quote timestamp skew
- Available quantity
- Capital lock duration
- Return on capital

It was important to take time into account, as there can be delays between the price being given and when the trade is actually executed, especially when there are multiple people in a virtual queue to do the same trade. This problem was far more prevalent during the initial single market arbitrage project, however became far less impactful here as opportunities are much more frequent.

## Paper-Trading Methodology

For the final test, I allowed the 26-market watchlist to be monitored for 4 hours. This is quite a short window considering the focus was to detect such rare occurrences; however, I was somewhat time-restricted due to difficulties developing the pair filtration system.

The main execution rules were as follows:

- Minimum raw ROC: 0.25%
- Minimum APR: 0.75%
- Minimum net profit per contract: $0.0025
- Maximum quote skew: 3 seconds
- Maximum contracts per position: 500
- Polling interval: 10 seconds

The entire watchlist was rescanned every 10 seconds (one cycle), with opportunities ranked primarily by annualized return on capital. Once a paper position was entered, its capital was treated as locked until the underlying contracts settled; it therefore remained locked at the end of the four-hour monitoring run.

## Results

A total of four trades were made, with three of them identified in the very first cycle, suggesting that those opportunities may already have been open before the monitoring run began.

| Market | Strategy | Entry APR | Raw ROC | Capital Deployed | Locked Paper Profit |
|---|---|---:|---:|---:|---:|
| Ajax | Kalshi YES + Polymarket NO | **3.04%** | **2.78%** | **$40.87** | **$1.13** |
| Zohran Mamdani | Kalshi YES + Polymarket NO | **1.30%** | **2.84%** | **$47.65** | **$1.35** |
| Mike Derry | Kalshi YES + Polymarket NO | **0.88%** | **0.91%** | **$4.95** | **$0.05** |
| Donald Trump Jr. | Kalshi NO + Polymarket YES | **0.77%** | **1.68%** | **$24.59** | **$0.41** |

| Portfolio Metric | Result |
|---|---:|
| Capital Deployed | **$118.05** |
| Locked Paper Profit | **$2.95** |
| Raw Return on Deployed Capital | **~2.50%** |

These final results were less profitable than expected. One likely factor is the strict nature of the semantic filters, which may have removed some genuine opportunities. A positive takeaway, however, is that the system substantially reduced the number of false-positive candidate pairs before execution.

![Trade-level return summary](figures/figure_D_trade_level_summary_fixed.png)

## Live Opportunity Behaviour

![APR across the final 26-market watchlist](figures/13_all_26_watchlist_apr_over_time.png)

The entire 26-market watchlist was monitored during the run, with only four markets ever crossing the 0.75% APR threshold. As most of these contracts described events not due to happen in the short term (such as 2028 elections), the prices were not extremely volatile during the four-hour run, due to a lack of market or media activity at this early stage. A much longer run of days or weeks would allow a higher threshold to be used as real-life events which relate to the contracts could cause large opportunities to appear.

![APR threshold crossings and near misses](figures/figure_A_threshold_crossings_near_misses.png)

Many of the events had no price change at all during the 4-hour window.

![APR over time with trade entries](figures/12_apr_over_time_trade_entries.png)

## Limitations

The biggest limitation was that the final system may have become too conservative. I intentionally prioritised precision because false-positive matches could create exposed risk in trades that appeared to be hedged. This led to the V7, V8, and V8.1 rule-based filters, with V8 in particular being highly selective and potentially removing a substantial number of genuine opportunities. Settlement verification added another conservative layer, reducing the 46 price-confirmed candidates to 26 watchlist markets in order to avoid contract-rule differences that could invalidate the hedge.

The live monitoring period lasted only 4 hours, so the observed opportunities cannot be assumed to represent normal conditions over longer periods. The final test also used simulated rather than personal capital. Live capital was not deployed because prediction-market access and compliance requirements vary by jurisdiction, and I wanted to keep the project focused on the technical system. Although the paper engine accounted for fees, quote skew, capital lock-up and some execution latency, real trading could introduce additional effects such as partial fills and operational error.

Finally, the annualized returns of the positions that passed every filter were relatively low. The project therefore did not demonstrate that this arbitrage approach could outperform more traditional investments. Instead, the result highlighted the central trade-off of the project: stronger protection against basis and settlement risk reduced the number of opportunities available to the strategy.

## Future Work

If I were to continue the project, I would focus on improving the efficiency of the filtration system, perhaps by splitting different categories into matching systems designed specifically for each type of contract. This could help address market-domain differences, as none of the Economics, Weather, or Technology categories had a single candidate make it to the final stage.

A second improvement would be to replace some of the binary pass/reject settlement decisions with probabilistic basis risk estimates. Instead of requiring a pair to be effectively risk-free, the engine could estimate the probability that a settlement difference becomes relevant and compare that risk against the expected return.

Other extensions I would consider include:

- training a settlement-aware classifier directly on full contract rules
- expanding the manually labelled equivalence dataset
- testing precision and recall on a larger independent dataset
- monitoring opportunities over days or weeks rather than hours
- introducing controlled risk budgets for near-equivalent contracts
- eventually testing small-scale live execution

Replacing the rule-based filtration with a precision-first learned model that works differently from the DeBERTa-based model would be a main goal for increasing opportunity coverage. A second precise model with different failure modes could provide another way to verify true pairs while catching false positives that slipped through the first model.

## Notes

### Open-Source Models

This project uses the following open-source pretrained models:

- **[DeBERTa V3 Base](https://huggingface.co/microsoft/deberta-v3-base)** by Microsoft, used as the base model for pairwise market-equivalence classification.
- **[BGE Small English v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)** by BAAI, used for high-recall semantic candidate retrieval.

Both models are released under the MIT License. Please refer to their original model cards and publications for full attribution and citation information.

### Requirements
- Python 3.x
- Project dependencies are listed in `requirements.txt`.
- Live market data requires access to the Kalshi and Polymarket APIs.

### Reproducing the Results
- Exact results may differ because market prices, liquidity, and available contracts change continuously.
- Large model checkpoints and market-data caches are excluded from the repository.
- The included notebook and saved outputs document the final experimental pipeline and paper-trading run.

### Paper Trading
- All trading results shown in this project are simulated.
- No personal capital was deployed.
- Prediction-market access and trading permissions vary by jurisdiction.
- Paper-trading performance should not be interpreted as guaranteed real-world profitability.

### Project Scope
- This project was developed for research and educational purposes.
- The system prioritises semantic equivalence, settlement compatibility, and execution realism over maximising the number of detected opportunities.

### License
- This project is licensed under the MIT License.
