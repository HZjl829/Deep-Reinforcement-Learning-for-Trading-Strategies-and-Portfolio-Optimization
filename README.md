# Deep RL based Order-Execution Model

Research code for a **deep-reinforcement-learning (DRL)** agent that schedules large equity orders for portfolio rebalancing—maximising expected wealth while minimising slippage, transaction costs, and risk.

## Highlights
- **Actor–Critic architecture (PyTorch)** trains on synthetic limit-order-book data and outputs time-weighted child orders.  
- **Market-dynamics simulator** generates realistic price/volume paths, boosting back-test robustness.  
- **Self-contained notebooks & scripts** for training (`train.py`), evaluation (`training+evaluation.ipynb`), and replication (`replicating.ipynb`).  
- Check-pointed models and performance **graphs/** included for quick inspection.
