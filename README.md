# 🔬 India LPG Supply Chain Resilience Stress Tester

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit 1.28+](https://img.shields.io/badge/streamlit-1.28+-green.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> GAN-powered disruption scenario generation and Monte Carlo stress testing for India's LPG supply chain network.

---

## 🎯 Overview

A comprehensive analytical framework that:
1. **Models the Network** - 20 nodes (sources, ports, refineries, bottling, markets), 26 edges
2. **Generates Scenarios** - Uses Conditional GAN trained on real PPAC & commodity data
3. **Runs Stress Tests** - 100+ Monte Carlo simulations for resilience analysis
4. **Calculates KPIs** - Fill rate, costs, vulnerability indices per node
5. **Visualizes Results** - Interactive Streamlit dashboard with India map and analytics

---

## ✨ Key Features

- **5 Disruption Types**: Red Sea, Refinery, Cyclone, Price Shock, Multi-Factor
- **Severity Control**: 0.1 (mild) to 1.0 (severe)
- **Season Effects**: Normal, Winter Peak, Monsoon
- **Monte Carlo**: 50-200 configurable simulations
- **Real Data**: GAN trained on historical disruption events (5000+ scenarios)
- **Max-Flow Analytics**: NetworkX optimal flow computation
- **Vulnerability Ranking**: Identify critical infrastructure nodes

---

## 🏗️ Architecture

```
Data Collection → GAN Training → Scenario Generation → Digital Twin → 
Max-Flow Analysis → KPI Calculation → Dashboard
```

---

## 📁 Project Structure

```
supply-chain-stress-tester/
├── data/
│   ├── nodes.csv                   # 20 network nodes
│   ├── edges.csv                   # 26 connections
│   └── training_scenarios.csv      # 5000+ training data
├── engine/
│   ├── digital_twin.py             # Network model + max-flow
│   └── stress_tester.py            # Batch testing
├── gan/
│   ├── data_collector.py           # Real data extraction
│   ├── model.py                    # Generator & Discriminator
│   ├── train.py                    # Training loop
│   └── generate.py                 # Scenario generation
├── app.py                          # Streamlit dashboard
├── tests/                          # Unit tests
└── models/saved_generator.pt       # Trained GAN weights
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/supply-chain-stress-tester.git
cd supply-chain-stress-tester
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Train GAN (First Run)

```bash
python -m gan.data_collector
python -m gan.train  # ~3-5 minutes
```

### Run Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📊 Usage

1. Select disruption type (Red Sea, Cyclone, etc.)
2. Set severity (0.1 - 1.0)
3. Choose season (Normal, Winter, Monsoon)
4. Configure runs (50-200)
5. Click "Run Stress Test"
6. View resilience score & vulnerable nodes ranking

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Network** | NetworkX (max-flow min-cost) |
| **GAN** | PyTorch (Conditional) |
| **Dashboard** | Streamlit + Plotly |
| **Data** | Pandas, NumPy, scikit-learn |
| **Prices** | yfinance |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test individual modules
pytest tests/test_digital_twin.py -v
pytest tests/test_stress_tester.py -v
pytest tests/test_gan.py -v
```

---

## ❗ Troubleshooting

| Issue | Solution |
|-------|----------|
| `max_flow_min_cost` error | Cast capacities to int in `digital_twin.py` |
| Fill rate < 80% | Increase edge capacities in `data/edges.csv` |
| GAN losses NaN | Reduce learning rate to `0.00005` |
| Import errors | Run from project root: `cd supply-chain-stress-tester` |
| Streamlit Cloud fails | Use CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

---

## 📈 Performance Benchmarks

| Operation | Time (CPU) | Notes |
|-----------|-----------|-------|
| Load network | <100ms | File I/O |
| Calculate KPI | ~500ms | Max-flow |
| Generate 100 scenarios | ~2s | GAN forward pass |
| Stress test (100 runs) | ~45s | Batch testing |
| Full dashboard | ~60s | End-to-end |

---

## 📝 License

MIT License - see LICENSE file

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📧 Support

- Open a GitHub Issue
- Check the Troubleshooting section
- Review `documentation.md` for detailed technical docs

---
