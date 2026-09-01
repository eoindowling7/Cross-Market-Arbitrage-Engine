# Cross-Market Prediction Arbitrage Engine
This is a precision first engine for identifying and trading cross-market arbitrage opportunities between Kalshi and Polymarket using a semantic pair filtration system, live pricing and execution conscious return filters.

![Candidate filtering pipeline](figures/02_candidate_filter_sankey_clean.png)

## Overview

This project focuses on identifying economically equivalent contracts listed on different prediction markets and then monitoring them in real time to detect any price changes which would cause arbitrage opportunities, guaranteeing a profit if done correctly.

The largest challenge in the project was certainly the contract matching, as a single false positive in pair evaluation meant a false arbitrage would likely undo any of the marginal profits made by genuine opportunities. Rigorous evaluation would be required to ensure any contracts matched were perfectly identical, beyond simply their titles but also including the platform specific clauses.

The system combines a trained semantic recognition model and layers deterministic precision filters for extremely high matching accuracy. These pairs are then checked for verification based on their settlement rules, with the passed pairs being checked for financial promise and watch-listed to be monitored over a longer cycle.

## Project Motivation

The idea originally began as a simple Kalshi-only arbitrage engine, and while genuine pricing errors could be detected they were rare and offered extremely low returns after fees. This form of arbitrage detection is also not viable in the long run as prediction markets may ban users placing such suspicious orders. This led me to using the other largest prediction market, polymarket, as a source for cross-market opportunities.

In theory, if I can buy opposing outcomes across the two platforms for less than the guaranteed settlement payout, the difference represents an arbitrage opportunity. However two contracts can look almost identical while differing in settlement date, event definition, cancellation rules, eligible outcomes, geographical scope, competition stage, or resolution procedure. This meant the project focus shifted away from simply finding price discrepancies, and towards building a system that could determine if two contracts were genuinely compatible enough to trade as a hedge.

## System Architecture

Although many designs were tested, this was the final architecture chosen for its near perfect pairing precision and relatively high recall.
## Market Matching Pipeline

### 1. Candidate Retrieval with BGE Embeddings
### 2. Pairwise Equivalence Classification with DeBERTa
### 3. Deterministic Semantic and Precision Filtering

## Settlement and Execution Filtering

### Settlement-Rule Verification
### Directional Payoff Equivalence
### Live Pricing and Execution Checks

## Paper-Trading Methodology

## Results

## Key Figures

## Limitations

## Future Work

## Repository Structure

## Setup and Usage

## Technologies Used
