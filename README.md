# Gas Pipeline Network Stress Tester

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gas-pipeline-stress-tester.streamlit.app/)

A comprehensive framework for stress testing US gas pipeline infrastructure using real-world data and generative AI. This tool combines the GEM-GGIT Global Gas Pipelines Dataset with a Conditional GAN trained on patterns from 15+ documented disruption events to identify critical infrastructure vulnerabilities.

## 🚀 Features

- **Real Pipeline Data**: Uses actual pipeline routes, capacities, and ownership from GEM-GGIT dataset
- **GAN-Generated Scenarios**: Conditional Generative Adversarial Network creates realistic disruption scenarios
- **Interactive Dashboard**: Streamlit-based visualization with geographic maps and vulnerability analysis
- **Network Analysis**: Graph-based modeling with connectivity, capacity, and betweenness centrality metrics
- **Corporate Risk Assessment**: Identifies ownership concentration risks in critical infrastructure

## 📊 Key Capabilities

- **Disruption Types**: Hurricane, sabotage, sanctions, demand surge, multi-factor combinations
- **Vulnerability Ranking**: Identifies single points of failure and critical hub nodes
- **Resilience Scoring**: Quantifies network robustness under various stress conditions
- **Geographic Visualization**: Interactive maps showing actual pipeline routes and hub locations
- **Ownership Analysis**: Reveals concentration risks from corporate control of pipeline capacity

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gas-pipeline-stress-tester.git
cd gas-pipeline-stress-tester

# Install dependencies
pip install -r requirements.txt

# Download the GEM-GGIT dataset
# Place GEM-GGIT-Gas-Pipelines-2025-11.geojson in data/raw/
```

### Data Processing
```bash
# Process the raw GeoJSON into network graph
python -m engine.data_processor

# This creates:
# - data/processed/nodes.csv (network nodes)
# - data/processed/edges.csv (pipeline connections)
```

### GAN Training
```bash
# Generate training data from real events
python -m gan.data_generator

# Train the Conditional GAN
python -m gan.train
```

## 🎯 Usage

### Local Dashboard
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

### Key Dashboard Features

1. **Pipeline Map**: Interactive geographic visualization of US gas pipeline network
2. **Stress Testing**: Configure disruption scenarios and run network simulations
3. **Vulnerability Analysis**: Ranked list of most critical infrastructure nodes
4. **Owner Concentration**: Analysis of corporate control over pipeline capacity

## 📈 Methodology

### Data Pipeline
1. **GeoJSON Processing**: Convert GEM-GGIT dataset to graph structure
2. **Geographic Clustering**: Identify hub nodes through endpoint clustering
3. **Network Modeling**: Create NetworkX graph with real capacities and routes

### GAN Training
- **Real Events**: 15+ documented disruptions (hurricanes, cyberattacks, sanctions)
- **Conditional Generation**: Type, severity, and regional targeting
- **Correlated Failures**: Captures geographic clustering patterns

### Stress Testing
- **Connectivity Analysis**: Network fragmentation under disruption
- **Capacity Assessment**: Total throughput retention
- **Vulnerability Ranking**: Impact scoring for individual nodes

## 📊 Sample Results

For a Category 3 hurricane targeting the Gulf Coast:
- **Resilience Score**: 67/100
- **Capacity Retained**: 45-78% (average 62%)
- **Network Fragmentation**: Up to 7 disconnected components
- **Critical Hubs**: 3-5 nodes whose failure causes >50% capacity loss

## 🏗️ Architecture

```
gas-pipeline-stress-tester/
├── data/
│   ├── raw/           # GEM-GGIT GeoJSON dataset
│   ├── processed/     # CSV files (nodes.csv, edges.csv)
│   └── training/      # GAN training scenarios
├── engine/
│   ├── data_processor.py  # GeoJSON → Graph conversion
│   ├── digital_twin.py    # Network simulation engine
│   └── stress_tester.py   # Vulnerability analysis
├── gan/
│   ├── data_generator.py  # Training data creation
│   ├── model.py          # GAN architecture
│   ├── train.py          # Training loop
│   └── generate.py       # Scenario generation
├── app.py               # Streamlit dashboard
└── requirements.txt
```

## 🔬 Technical Details

### Network Model
- **Nodes**: Infrastructure hubs, junctions, terminals (clustered from pipeline endpoints)
- **Edges**: Pipeline segments with real capacity and length data
- **Metrics**: Connectivity, capacity, betweenness centrality, edge connectivity

### GAN Architecture
- **Generator**: 3-layer MLP with conditional input (type, severity, region)
- **Discriminator**: 3-layer MLP for realism assessment
- **Training**: 500 epochs on 4,500+ augmented scenarios
- **Output**: 11-dimensional disruption vectors (regional multipliers)

### Disruption Types
- **Hurricane**: Geographically clustered coastal impact
- **Sabotage**: Targeted corridor shutdowns
- **Sanctions**: Sustained capacity reductions
- **Demand Surge**: Capacity exceeded by demand
- **Multi-Factor**: Combined disruption scenarios

## 📚 Data Sources

- **GEM-GGIT Global Gas Pipelines Dataset**: Global Energy Monitor (2025)
- **Henry Hub Natural Gas Prices**: Yahoo Finance historical data
- **Disruption Events**: 15+ documented real-world incidents (2005-2023)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for research and educational purposes. Pipeline infrastructure data and analysis results should not be used for operational decisions without validation by qualified engineering professionals. The authors assume no liability for the use or misuse of this software.

## 🙏 Acknowledgments

- Global Energy Monitor for the GEM-GGIT dataset
- PyTorch team for the deep learning framework
- Streamlit team for the dashboard framework
- NetworkX developers for graph analysis tools

---

**Built with ❤️ for energy infrastructure resilience research**
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
