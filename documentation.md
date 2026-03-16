# 🔬 India LPG Supply Chain Stress Tester
## Complete Execution Plan & Documentation

---

# TABLE OF CONTENTS

```
1. PROJECT OVERVIEW
2. TECH STACK & PREREQUISITES
3. REPOSITORY STRUCTURE
4. PHASE-BY-PHASE EXECUTION PLAN
   Phase 1: Environment Setup & GitHub Init
   Phase 2: Data Layer
   Phase 3: Digital Twin Engine
   Phase 4: Stress Testing Engine
   Phase 5: GAN Data Collection
   Phase 6: GAN Model & Training
   Phase 7: GAN Generation & Validation
   Phase 8: Streamlit Dashboard
   Phase 9: Integration & Testing
   Phase 10: Deployment & Documentation
5. FILE-BY-FILE IMPLEMENTATION GUIDE
6. TESTING CHECKPOINTS
7. GITHUB WORKFLOW
8. TROUBLESHOOTING
9. README.md TEMPLATE
```

---

# 1. PROJECT OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│  PROJECT: India LPG Supply Chain Resilience Stress Tester   │
│                                                             │
│  WHAT IT DOES:                                              │
│  1. Models India's LPG supply chain as a network graph      │
│  2. Uses a GAN trained on REAL data to generate disruption  │
│     scenarios (Red Sea crisis, cyclones, refinery shutdowns) │
│  3. Runs Monte Carlo stress tests (100 simulations)         │
│  4. Calculates resilience score + vulnerability ranking     │
│  5. Displays everything in an interactive Streamlit dashboard│
│                                                             │
│  PIPELINE:                                                  │
│  Real Data → GAN Training → Scenario Generation →           │
│  Digital Twin → Max-Flow Analysis → KPI Calculation →       │
│  Dashboard Visualization                                    │
└─────────────────────────────────────────────────────────────┘
```

---

# 2. TECH STACK & PREREQUISITES

## Software Required

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Core language |
| VSCode | Latest | IDE |
| Git | Latest | Version control |
| GitHub Account | - | Repository hosting |

## Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 2.0+ | GAN model |
| numpy | 1.24+ | Numerical computation |
| pandas | 2.0+ | Data handling |
| networkx | 3.1+ | Graph modeling & max-flow |
| plotly | 5.15+ | Interactive visualizations |
| streamlit | 1.28+ | Dashboard framework |
| scikit-learn | 1.3+ | Data preprocessing |
| yfinance | 0.2+ | Real price data download |
| matplotlib | 3.7+ | Static plots (backup) |

## VSCode Extensions (Recommended)

```
- Python (Microsoft)
- Pylance
- GitLens
- Streamlit (optional)
- CSV Rainbow (for viewing CSV files)
```

---

# 3. REPOSITORY STRUCTURE

```
supply-chain-stress-tester/
│
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                          ← Streamlit dashboard (entry point)
│
├── data/
│   ├── nodes.csv                   ← Network nodes (20 nodes)
│   ├── edges.csv                   ← Network edges (26 edges)
│   └── training_scenarios.csv      ← GAN training data (generated)
│
├── engine/
│   ├── __init__.py
│   ├── digital_twin.py             ← Network model + max-flow
│   └── stress_tester.py            ← Batch testing + vulnerability
│
├── gan/
│   ├── __init__.py
│   ├── data_collector.py           ← Real data collection + augmentation
│   ├── model.py                    ← Generator + Discriminator architecture
│   ├── train.py                    ← Training loop
│   └── generate.py                 ← Scenario generation + validation
│
├── models/
│   └── saved_generator.pt          ← Trained GAN weights
│
├── tests/
│   ├── test_digital_twin.py        ← Unit tests for twin
│   ├── test_stress_tester.py       ← Unit tests for stress tester
│   └── test_gan.py                 ← Unit tests for GAN
│
└── notebooks/                      ← (Optional) Jupyter exploration
    └── exploration.ipynb
```

---

# 4. PHASE-BY-PHASE EXECUTION PLAN

---

## PHASE 1: Environment Setup & GitHub Repository Init

**Duration:** 30-45 minutes
**Goal:** Set up local dev environment, initialize Git repo, push initial structure to GitHub

### Module 1.1: Create GitHub Repository

**Steps:**
```
Step 1.1.1: Go to github.com → Click "+" → "New repository"
Step 1.1.2: Name: "supply-chain-stress-tester"
Step 1.1.3: Description: "GAN-powered stress testing of India's LPG supply chain"
Step 1.1.4: Set to Public
Step 1.1.5: Check "Add a README file"
Step 1.1.6: Add .gitignore → Select "Python"
Step 1.1.7: Click "Create repository"
Step 1.1.8: Copy the repository URL (HTTPS)
```

### Module 1.2: Clone & Setup in VSCode

**Steps:**
```
Step 1.2.1: Open VSCode
Step 1.2.2: Open Terminal (Ctrl + `)
Step 1.2.3: Navigate to your projects folder:
            cd ~/Projects   (or wherever you keep projects)
Step 1.2.4: Clone the repo:
            git clone https://github.com/YOUR_USERNAME/supply-chain-stress-tester.git
Step 1.2.5: Open the cloned folder in VSCode:
            cd supply-chain-stress-tester
            code .
```

### Module 1.3: Create Folder Structure

**Steps:**
```
Step 1.3.1: In VSCode terminal, run these commands:

mkdir data
mkdir engine
mkdir gan
mkdir models
mkdir tests
mkdir notebooks

touch engine/__init__.py
touch gan/__init__.py
touch engine/digital_twin.py
touch engine/stress_tester.py
touch gan/data_collector.py
touch gan/model.py
touch gan/train.py
touch gan/generate.py
touch app.py
touch tests/test_digital_twin.py
touch tests/test_stress_tester.py
touch tests/test_gan.py
```

### Module 1.4: Create Virtual Environment & Install Dependencies

**Steps:**
```
Step 1.4.1: Create virtual environment:
            python -m venv venv

Step 1.4.2: Activate it:
            Windows: venv\Scripts\activate
            Mac/Linux: source venv/bin/activate

Step 1.4.3: Create requirements.txt with this content:
            (see file content in Section 5)

Step 1.4.4: Install all packages:
            pip install -r requirements.txt

Step 1.4.5: Verify installation:
            python -c "import torch, networkx, streamlit, plotly, pandas; print('All good!')"
```

### Module 1.5: Initial Git Commit

**Steps:**
```
Step 1.5.1: Update .gitignore to add:
            venv/
            __pycache__/
            *.pyc
            .env
            models/*.pt
            data/training_scenarios.csv

Step 1.5.2: Stage all files:
            git add .

Step 1.5.3: Commit:
            git commit -m "feat: initial project structure with folder layout"

Step 1.5.4: Push:
            git push origin main
```

**✅ Checkpoint:** GitHub repo shows folder structure with empty files.

---

## PHASE 2: Data Layer

**Duration:** 1-1.5 hours
**Goal:** Create and validate the CSV data files that define the supply chain network

### Module 2.1: Create nodes.csv

**File:** `data/nodes.csv`

**What this file contains:**
```
- 20 nodes representing the India LPG supply chain
- 3 international sources (Saudi Arabia, UAE, Qatar)
- 3 import ports (Kandla, JNPT Mumbai, Vizag)
- 3 domestic refineries (Jamnagar, Mumbai, Paradip)
- 4 bottling plant clusters (North, West, South, East India)
- 6 market regions (Delhi NCR, UP, Maharashtra, Tamil Nadu, West Bengal, Gujarat)
- ALL units in MT/day (metric tons per day)
- demand_mt_day is SEPARATE from capacity_mt_day
```

**Implementation Instructions:**
```
Step 2.1.1: Open data/nodes.csv in VSCode
Step 2.1.2: Copy the exact CSV content provided in Section 5
Step 2.1.3: Save the file
Step 2.1.4: Verify: Open in terminal with
            python -c "import pandas as pd; df=pd.read_csv('data/nodes.csv'); print(df.shape); print(df.head())"
            Expected: (20, 9)
```

### Module 2.2: Create edges.csv

**File:** `data/edges.csv`

**What this file contains:**
```
- 26 directed edges (no duplicates, no backwards flow)
- Flow: Source→Port→Bottling→Market AND Refinery→Bottling→Market
- Each edge has: capacity (MT/day), cost (₹/MT), transit time (days), reliability
- NO R→P edges (refineries connect directly to bottling clusters)
- NO duplicate edges between same node pair
```

**Implementation Instructions:**
```
Step 2.2.1: Open data/edges.csv in VSCode
Step 2.2.2: Copy the exact CSV content provided in Section 5
Step 2.2.3: Save the file
Step 2.2.4: Verify: Open in terminal with
            python -c "import pandas as pd; df=pd.read_csv('data/edges.csv'); print(df.shape); print(df[['source','target']].values)"
            Expected: (26, 8)
```

### Module 2.3: Data Validation Script

**File:** Create a quick validation (run once, don't need to keep)

**Implementation Instructions:**
```
Step 2.3.1: Open VSCode terminal
Step 2.3.2: Run this Python snippet to validate:

python -c "
import pandas as pd
nodes = pd.read_csv('data/nodes.csv')
edges = pd.read_csv('data/edges.csv')

# Check 1: All edge sources/targets exist in nodes
node_ids = set(nodes['id'].values)
for _, row in edges.iterrows():
    assert row['source'] in node_ids, f'Missing source: {row[\"source\"]}'
    assert row['target'] in node_ids, f'Missing target: {row[\"target\"]}'
print('✓ All edge endpoints exist in nodes')

# Check 2: No duplicate edges
edge_pairs = edges[['source','target']].apply(tuple, axis=1)
assert edge_pairs.nunique() == len(edges), 'Duplicate edges found!'
print('✓ No duplicate edges')

# Check 3: Supply > Demand
total_supply = nodes[nodes['type'].isin(['source','refinery'])]['capacity_mt_day'].sum()
total_demand = nodes[nodes['type']=='market']['demand_mt_day'].sum()
print(f'✓ Total supply: {total_supply} MT/day')
print(f'✓ Total demand: {total_demand} MT/day')
print(f'✓ Supply/Demand ratio: {total_supply/total_demand:.2f}x')

# Check 4: Markets have enough inflow capacity
for _, m in nodes[nodes['type']=='market'].iterrows():
    inflow = edges[edges['target']==m['id']]['capacity_mt_day'].sum()
    print(f'  {m[\"id\"]}: demand={m[\"demand_mt_day\"]}, inflow_capacity={inflow}, ratio={inflow/m[\"demand_mt_day\"]:.2f}x')
print('✓ All checks passed!')
"
```

### Module 2.4: Git Commit for Data

```
Step 2.4.1: git add data/nodes.csv data/edges.csv
Step 2.4.2: git commit -m "feat: add validated supply chain network data (nodes + edges)"
Step 2.4.3: git push origin main
```

**✅ Checkpoint:** Both CSV files validated. Supply/demand ratio ≈ 1.75x. All markets have sufficient inflow.

---

## PHASE 3: Digital Twin Engine

**Duration:** 2-3 hours
**Goal:** Build the core network model that loads CSVs, builds a NetworkX DiGraph, and computes optimal flow using max-flow algorithm

### Module 3.1: Create engine/__init__.py

**File:** `engine/__init__.py`

**Implementation Instructions:**
```
Step 3.1.1: Open engine/__init__.py
Step 3.1.2: Add a one-line docstring (see Section 5)
Step 3.1.3: Save
```

### Module 3.2: Build DigitalTwin class — Constructor & Load

**File:** `engine/digital_twin.py`

**What this class does:**
```
- __init__: Initialize empty attributes
- load_network: Read CSVs → Build NetworkX DiGraph with proper attributes
- Validation: Ensure all edge endpoints exist, no duplicates
- Store original graph copy for reset functionality
```

**Implementation Instructions:**
```
Step 3.2.1: Open engine/digital_twin.py in VSCode
Step 3.2.2: Write imports: pandas, networkx, numpy, copy
Step 3.2.3: Create class DigitalTwin
Step 3.2.4: Write __init__ method:
            - self.graph = None
            - self.original_graph = None
            - self.nodes_df = None
            - self.edges_df = None

Step 3.2.5: Write load_network(self, nodes_path, edges_path) method:
            - Read both CSVs with pd.read_csv
            - Create nx.DiGraph()
            - Loop through nodes_df rows: add_node with all attributes
            - Loop through edges_df rows: add_edge with:
              * capacity = capacity_mt_day × reliability (effective capacity)
              * weight = cost_per_mt (NetworkX max-flow uses 'weight' for cost)
              * Store original_capacity and original_weight for reset
            - Validate: check all edge endpoints exist in node set
            - Deep copy graph as self.original_graph

Step 3.2.6: Test it immediately:
            python -c "
            from engine.digital_twin import DigitalTwin
            twin = DigitalTwin()
            twin.load_network('data/nodes.csv', 'data/edges.csv')
            print(f'Nodes: {twin.graph.number_of_nodes()}')
            print(f'Edges: {twin.graph.number_of_edges()}')
            "
            Expected: Nodes: 20, Edges: 26
```

### Module 3.3: Build calculate_kpis method

**What this method does:**
```
- Creates a copy of the graph
- Adds SUPER_SOURCE connected to all source + refinery nodes
- Adds SUPER_SINK connected to all market nodes (capacity = demand)
- Runs nx.max_flow_min_cost() to find optimal flow
- Extracts: fill_rate, total_cost, total_flow, per_market_fill
```

**Implementation Instructions:**
```
Step 3.3.1: In digital_twin.py, add calculate_kpis(self) method
Step 3.3.2: Create a working copy of self.graph
Step 3.3.3: Add "SUPER_SOURCE" node
Step 3.3.4: For each source/refinery node: add edge from SUPER_SOURCE
            with capacity = node's capacity_mt_day × reliability
            and weight = 0
Step 3.3.5: Add "SUPER_SINK" node
Step 3.3.6: For each market node: add edge to SUPER_SINK
            with capacity = demand_mt_day and weight = 0
Step 3.3.7: Call nx.max_flow_min_cost(G, "SUPER_SOURCE", "SUPER_SINK")
Step 3.3.8: Calculate total_flow = sum of flows into SUPER_SINK
Step 3.3.9: Calculate fill_rate = (total_flow / total_demand) × 100
Step 3.3.10: Calculate total_cost by summing flow × weight for each used edge
Step 3.3.11: Calculate per-market fill rates
Step 3.3.12: Return dict with all KPIs

Step 3.3.13: IMPORTANT - if max_flow_min_cost errors:
             Cast all capacities and weights to int:
             capacity=int(effective_cap), weight=int(cost)
             
Step 3.3.14: Test:
             python -c "
             from engine.digital_twin import DigitalTwin
             twin = DigitalTwin()
             twin.load_network('data/nodes.csv', 'data/edges.csv')
             kpis = twin.calculate_kpis()
             print(f'Fill rate: {kpis[\"fill_rate\"]:.1f}%')
             print(f'Total flow: {kpis[\"total_flow\"]:.0f} MT/day')
             print(f'Total cost: ₹{kpis[\"total_cost\"]:,.0f}')
             "
             Expected: Fill rate should be 85-95%
```

### Module 3.4: Build apply_disruption and reset methods

**What these methods do:**
```
- apply_disruption: Takes a disruption dict, modifies edge capacities/costs in the graph
- reset: Restores graph to original state from deep copy
```

**Implementation Instructions:**
```
Step 3.4.1: Write apply_disruption(self, disruption) method
            disruption format:
            {
              "node_multipliers": {"P1": 0.3, "R1": 0.5},
              "edge_cost_multipliers": {("S1","P1"): 3.0},
              "edge_capacity_multipliers": {("S1","P1"): 0.4}
            }

Step 3.4.2: For node_multipliers: find all edges connected to that node
            and multiply their capacity by the multiplier

Step 3.4.3: For edge_cost_multipliers: multiply the edge's 'weight'

Step 3.4.4: For edge_capacity_multipliers: multiply the edge's 'capacity'

Step 3.4.5: Write reset(self) method:
            self.graph = copy.deepcopy(self.original_graph)

Step 3.4.6: Test:
            python -c "
            from engine.digital_twin import DigitalTwin
            twin = DigitalTwin()
            twin.load_network('data/nodes.csv', 'data/edges.csv')
            baseline = twin.calculate_kpis()
            print(f'Baseline: {baseline[\"fill_rate\"]:.1f}%')
            twin.apply_disruption({'node_multipliers': {'P1': 0.2}})
            stressed = twin.calculate_kpis()
            print(f'Stressed: {stressed[\"fill_rate\"]:.1f}%')
            twin.reset()
            restored = twin.calculate_kpis()
            print(f'Restored: {restored[\"fill_rate\"]:.1f}%')
            "
            Expected: Stressed < Baseline, Restored ≈ Baseline
```

### Module 3.5: Git Commit

```
git add engine/
git commit -m "feat: digital twin with max-flow KPI calculation"
git push origin main
```

**✅ Checkpoint:** DigitalTwin loads network, baseline fill rate 85-95%, disruptions cause measurable drops, reset works.

---

## PHASE 4: Stress Testing Engine

**Duration:** 1-1.5 hours
**Goal:** Build batch stress testing and vulnerability analysis functions

### Module 4.1: Build run_batch_test function

**File:** `engine/stress_tester.py`

**What this function does:**
```
- Takes a DigitalTwin instance and a list of disruption scenarios
- For each scenario: apply → calculate KPIs → reset
- Collects fill_rate, cost, fill_rate_drop, cost_increase_pct per scenario
- Returns list of result dicts
```

**Implementation Instructions:**
```
Step 4.1.1: Open engine/stress_tester.py
Step 4.1.2: Import numpy, pandas
Step 4.1.3: Write run_batch_test(twin, scenarios) function:
            - Get baseline KPIs once
            - Loop through scenarios:
              * twin.apply_disruption(scenario)
              * stressed = twin.calculate_kpis()
              * Compute fill_rate_drop and cost_increase_pct
              * Append to results list
              * twin.reset()
            - Return results list
```

### Module 4.2: Build find_vulnerable_nodes function

**What this function does:**
```
- Tests each non-market node individually
- Applies 90% capacity reduction to each node
- Measures fill rate drop
- Returns ranked list of (node_id, impact_score) tuples
```

**Implementation Instructions:**
```
Step 4.2.1: Write find_vulnerable_nodes(twin) function
Step 4.2.2: Get baseline
Step 4.2.3: Loop through each node where type != "market"
Step 4.2.4: Apply disruption {node_multipliers: {node: 0.1}}
Step 4.2.5: Calculate stressed KPIs
Step 4.2.6: Record fill_rate_drop
Step 4.2.7: Reset
Step 4.2.8: Sort by impact (highest first)
Step 4.2.9: Return sorted list
```

### Module 4.3: Build find_critical_edges function

**What this function does:**
```
- Same as vulnerable nodes but for edges
- Tests each edge individually with 90% capacity reduction
```

**Implementation Instructions:**
```
Step 4.3.1: Write find_critical_edges(twin) function
Step 4.3.2: Same logic as nodes but using edge_capacity_multipliers
```

### Module 4.4: Test Stress Tester

```
Step 4.4.1: Run:
python -c "
from engine.digital_twin import DigitalTwin
from engine.stress_tester import find_vulnerable_nodes

twin = DigitalTwin()
twin.load_network('data/nodes.csv', 'data/edges.csv')
vuln = find_vulnerable_nodes(twin)
print('Top 5 vulnerable nodes:')
for node, score in vuln[:5]:
    print(f'  {node}: {score:.1f}% fill rate drop')
"
Expected: Ports (P1, P2) should rank highest
```

### Module 4.5: Git Commit

```
git add engine/stress_tester.py
git commit -m "feat: stress tester with batch testing and vulnerability analysis"
git push origin main
```

**✅ Checkpoint:** Batch test runs 100 scenarios in <30 sec. Vulnerability ranking has ports at top.

---

## PHASE 5: GAN Data Collection

**Duration:** 2-3 hours
**Goal:** Collect real disruption data and create augmented training dataset

### Module 5.1: Download Real Price Data

**File:** `gan/data_collector.py`

**What this does:**
```
- Uses yfinance to download 10 years of Brent crude oil and USD/INR prices
- Calculates monthly deviation ratios from 12-month moving average
- Identifies months with >15% deviation as price disruption events
```

**Implementation Instructions:**
```
Step 5.1.1: Open gan/data_collector.py
Step 5.1.2: Import: yfinance, pandas, numpy, json
Step 5.1.3: Write download_price_data() function:
            - Download Brent crude ("BZ=F") from 2015-01-01
            - Download USD/INR ("INR=X") from 2015-01-01
            - Resample to monthly means
            - Calculate ratio vs 12-month rolling average
            - Return DataFrame with date, oil_ratio, inr_ratio
Step 5.1.4: Test:
            python -c "from gan.data_collector import download_price_data; print(download_price_data().head())"
```

### Module 5.2: Create Historical Event Vectors

**What this does:**
```
- Manually defines 15-20 major real disruption events (2015-2024)
- Each event is a dict with capacity multipliers for each node
- Events include: COVID, Suez blockage, Red Sea 2024, Cyclone Biparjoy,
  oil price crashes, refinery fires, etc.
```

**Implementation Instructions:**
```
Step 5.2.1: Write create_historical_events() function
Step 5.2.2: Create a list of dicts, each representing a real event
Step 5.2.3: Each dict has:
            - name, type, severity, region
            - P1_cap through B4_cap (13 node capacity multipliers)
            - sea_cost_mult, sea_transit_mult
            - road_cap_mult, rail_cap_mult, demand_mult
Step 5.2.4: Create at least 15 events (see Section 5 for full list)
```

### Module 5.3: Augment Data with Variations

**What this does:**
```
- For each real event, creates 250 variations with random noise
- Adds Gaussian noise (σ=0.08) to each numerical field
- Clamps values to valid ranges
- Results in 5000+ training samples
```

**Implementation Instructions:**
```
Step 5.3.1: Write augment_events(events, variations_per_event=250) function
Step 5.3.2: For each event, create variations with np.random.normal noise
Step 5.3.3: Clamp capacity multipliers to [0, 1]
Step 5.3.4: Clamp cost/transit multipliers to [1.0, 5.0]
Step 5.3.5: Return list of all augmented scenarios
```

### Module 5.4: Combine & Save Training Dataset

**What this does:**
```
- Combines price-derived scenarios with historical event variations
- Normalizes all values to 0-1 range for GAN training
- Saves as data/training_scenarios.csv
```

**Implementation Instructions:**
```
Step 5.4.1: Write build_training_dataset() function
Step 5.4.2: Call download_price_data() — convert price disruptions to scenarios
Step 5.4.3: Call create_historical_events()
Step 5.4.4: Call augment_events() on both
Step 5.4.5: Combine into single DataFrame
Step 5.4.6: Save to data/training_scenarios.csv
Step 5.4.7: Print stats: total rows, columns, value ranges
Step 5.4.8: Run it: python -m gan.data_collector
            Expected: "Saved 5000+ scenarios to data/training_scenarios.csv"
```

### Module 5.5: Git Commit

```
git add gan/data_collector.py
git commit -m "feat: GAN training data from real PPAC + price data"
git push origin main
```

**✅ Checkpoint:** training_scenarios.csv has 5000+ rows, ~20 columns, all values in valid ranges.

---

## PHASE 6: GAN Model & Training

**Duration:** 2-3 hours
**Goal:** Build Generator and Discriminator architectures, train the GAN

### Module 6.1: Build GAN Architecture

**File:** `gan/model.py`

**Architecture Summary:**
```
Generator:
  Input: z(32) + condition(7) = 39
  → Linear(39, 128) → LeakyReLU → BatchNorm
  → Linear(128, 256) → LeakyReLU → BatchNorm
  → Linear(256, 128) → LeakyReLU → BatchNorm
  → Linear(128, 18) → Sigmoid
  Output: 18 values in [0, 1]

Discriminator:
  Input: x(18) + condition(7) = 25
  → Linear(25, 128) → LeakyReLU → Dropout(0.3)
  → Linear(128, 64) → LeakyReLU → Dropout(0.3)
  → Linear(64, 1) → Sigmoid
  Output: probability (real/fake)

Condition vector (7 dims):
  - type_onehot: 5 dims (red_sea, refinery, cyclone, price_shock, multi)
  - severity: 1 dim (0-1)
  - region_encoded: 1 dim (0=all, 0.33=west, 0.66=east, 1=south)

Output vector (18 dims):
  - 13 node capacity multipliers (S1-S3, P1-P3, R1-R3, B1-B4)
  - sea_cost_mult, sea_transit_mult
  - road_cap_mult, rail_cap_mult
  - demand_mult
```

**Implementation Instructions:**
```
Step 6.1.1: Open gan/model.py
Step 6.1.2: Import torch, torch.nn
Step 6.1.3: Define class Generator(nn.Module):
            - __init__: define 4 linear layers with activation + batchnorm
            - forward(self, z, condition): concatenate z+c, pass through layers
Step 6.1.4: Define class Discriminator(nn.Module):
            - __init__: define 3 linear layers with activation + dropout
            - forward(self, x, condition): concatenate x+c, pass through layers
Step 6.1.5: Test:
            python -c "
            from gan.model import Generator, Discriminator
            import torch
            G = Generator(z_dim=32, c_dim=7, output_dim=18)
            D = Discriminator(input_dim=18, c_dim=7)
            z = torch.randn(4, 32)
            c = torch.randn(4, 7)
            fake = G(z, c)
            print(f'Generator output shape: {fake.shape}')  # (4, 18)
            score = D(fake, c)
            print(f'Discriminator output shape: {score.shape}')  # (4, 1)
            "
```

### Module 6.2: Build Training Loop

**File:** `gan/train.py`

**What this does:**
```
- Loads training_scenarios.csv
- Prepares condition vectors and target vectors
- Normalizes data with MinMaxScaler
- Trains GAN for 500 epochs with:
  * Label smoothing (real=0.9, fake=0.1)
  * Gradient clipping (max_norm=1.0)
  * Adam optimizer (lr=0.0002, betas=(0.5, 0.999))
- Saves trained generator to models/saved_generator.pt
```

**Implementation Instructions:**
```
Step 6.2.1: Open gan/train.py
Step 6.2.2: Import: torch, pandas, numpy, sklearn.preprocessing
Step 6.2.3: Write data loading function:
            - Load training_scenarios.csv
            - Separate condition columns (type, severity, region)
            - Separate feature columns (the 18 output dims)
            - One-hot encode disruption type → 5 dims
            - Normalize features with MinMaxScaler
            - Convert to torch tensors
            - Create DataLoader with batch_size=64

Step 6.2.4: Write training loop:
            - Initialize G, D, optimizers
            - For each epoch (500 total):
              For each batch:
                # Train Discriminator
                - Generate fake samples from G
                - D scores real (label=0.9) and fake (label=0.1)
                - Compute BCE loss
                - Clip gradients (max_norm=1.0)
                - Update D
                
                # Train Generator
                - Generate fake samples
                - D scores fake (want label=0.9)
                - Compute BCE loss
                - Clip gradients
                - Update G
              
              Print losses every 50 epochs

Step 6.2.5: Save generator:
            torch.save(G.state_dict(), "models/saved_generator.pt")
            Also save the MinMaxScaler for denormalization

Step 6.2.6: Run training:
            python -m gan.train
            Expected: Losses converge, no NaN, takes 1-3 min on CPU
```

### Module 6.3: Git Commit

```
git add gan/model.py gan/train.py
git commit -m "feat: conditional GAN architecture and training loop"
git push origin main
```

**✅ Checkpoint:** GAN trains without errors. Losses are not NaN. Generator weights saved.

---

## PHASE 7: GAN Generation & Validation

**Duration:** 1.5-2 hours
**Goal:** Build scenario generation pipeline with post-processing validation

### Module 7.1: Build Scenario Generator

**File:** `gan/generate.py`

**What this does:**
```
- Loads trained generator weights
- Takes disruption type, severity, region as input
- Generates N scenarios by sampling random noise vectors
- Denormalizes GAN output to actual multiplier ranges
- Validates each scenario (physical constraints)
- Converts to disruption dict format compatible with DigitalTwin
```

**Implementation Instructions:**
```
Step 7.1.1: Open gan/generate.py
Step 7.1.2: Write encode_condition(type, severity, region) function
Step 7.1.3: Write denormalize(output_tensor) function:
            - Node caps: output[0:13] → already 0-1 (capacity fraction)
            - sea_cost_mult: 1 + output[13] * 4 → range [1.0, 5.0]
            - sea_transit_mult: 1 + output[14] * 4 → range [1.0, 5.0]
            - road_cap_mult: output[15] → 0-1
            - rail_cap_mult: output[16] → 0-1
            - demand_mult: 0.5 + output[17] * 1.0 → range [0.5, 1.5]

Step 7.1.4: Write validate_scenario(scenario, disruption_type, region):
            - Red Sea: domestic nodes can't drop below 70%
            - Cyclone: only target region affected
            - Minimum total supply: at least 30% must remain
            - Road transport unaffected by shipping disruptions
            - Clamp all values to physical ranges

Step 7.1.5: Write generate_scenarios(type, severity, region, n=100):
            - Load generator from models/saved_generator.pt
            - Encode condition vector
            - Loop n times:
              * Sample z from N(0,1)
              * Forward through generator
              * Denormalize
              * Validate
              * Convert to disruption dict format
            - Return list of disruption dicts

Step 7.1.6: Write convert_to_disruption_dict(scenario) function:
            Maps the 18 values to the format DigitalTwin expects:
            {
              "node_multipliers": {"S1": val, "P1": val, ...},
              "edge_cost_multipliers": {("S1","P1"): val, ...},
              "edge_capacity_multipliers": {...}
            }

Step 7.1.7: Test:
            python -c "
            from gan.generate import generate_scenarios
            scenarios = generate_scenarios('red_sea', 0.8, 'all', 5)
            for i, s in enumerate(scenarios):
                print(f'Scenario {i}: {s}')
            "
            Expected: 5 different (not identical!) disruption dicts
```

### Module 7.2: End-to-End Pipeline Test

```
Step 7.2.1: Create a test script (or run in terminal):

python -c "
from engine.digital_twin import DigitalTwin
from engine.stress_tester import run_batch_test, find_vulnerable_nodes
from gan.generate import generate_scenarios
import numpy as np

# Load
twin = DigitalTwin()
twin.load_network('data/nodes.csv', 'data/edges.csv')
baseline = twin.calculate_kpis()
print(f'Baseline fill rate: {baseline[\"fill_rate\"]:.1f}%')

# Generate
scenarios = generate_scenarios('red_sea', 0.8, 'all', 20)
print(f'Generated {len(scenarios)} scenarios')

# Test
results = run_batch_test(twin, scenarios)
avg_fill = np.mean([r['fill_rate'] for r in results])
print(f'Avg stressed fill rate: {avg_fill:.1f}%')
print(f'Resilience score: {avg_fill:.0f}/100')

# Vulnerability
vuln = find_vulnerable_nodes(twin)
print('Top 5 vulnerable:')
for n, s in vuln[:5]:
    print(f'  {n}: {s:.1f}% drop')
"
```

### Module 7.3: Git Commit

```
git add gan/generate.py
git commit -m "feat: GAN scenario generation with validation"
git push origin main
```

**✅ Checkpoint:** Full pipeline works: GAN generates → Twin runs → KPIs calculated. Numbers are reasonable.

---

## PHASE 8: Streamlit Dashboard

**Duration:** 3-4 hours
**Goal:** Build the interactive web dashboard with all visualizations

### Module 8.1: Basic App Structure

**File:** `app.py`

**Implementation Instructions:**
```
Step 8.1.1: Open app.py
Step 8.1.2: Write imports:
            streamlit, plotly.graph_objects, pandas, numpy,
            and all project modules

Step 8.1.3: Set page config:
            st.set_page_config(
              page_title="India LPG Supply Chain Stress Tester",
              page_icon="🔬",
              layout="wide"
            )

Step 8.1.4: Add title and description:
            st.title("🔬 India LPG Supply Chain Stress Tester")
            st.caption("GAN-powered resilience analysis...")

Step 8.1.5: Load network with caching:
            @st.cache_resource
            def load_twin():
                twin = DigitalTwin()
                twin.load_network(...)
                return twin

Step 8.1.6: Test: streamlit run app.py
            Expected: Browser opens, title shows, no errors
```

### Module 8.2: Sidebar Controls

**Implementation Instructions:**
```
Step 8.2.1: Add sidebar header
Step 8.2.2: Add disruption type selectbox (5 options)
Step 8.2.3: Add severity slider (0.1 to 1.0)
Step 8.2.4: Add season selectbox (Normal, Winter, Monsoon)
Step 8.2.5: Add Monte Carlo runs slider (50-200, default 100)
Step 8.2.6: Add RUN button (type="primary")
Step 8.2.7: Add "About" expander with project description
```

### Module 8.3: Tab 1 — Network View

**Implementation Instructions:**
```
Step 8.3.1: Create two tabs: "🗺️ Network View" and "📊 Results"
Step 8.3.2: In tab1, build Plotly Scattergeo map:
            - Draw edges as lines (gray)
            - Draw nodes as colored dots:
              source=red, port=blue, refinery=orange,
              bottling=green, market=purple
            - Center on India (lat=22, lon=78)
            - Use scope="asia" with projection_scale=4

Step 8.3.3: Add baseline KPI metrics below the map:
            - 3 columns: Fill Rate, Daily Cost, Network Stats

Step 8.3.4: Add node data table (expandable):
            st.expander("View Network Data")
              st.dataframe(nodes_df)

Step 8.3.5: Test: streamlit run app.py
            Expected: Map shows India with nodes and edges
```

### Module 8.4: Tab 2 — Stress Test Results

**Implementation Instructions:**
```
Step 8.4.1: Handle the RUN button click:
            if run_button:
              with st.spinner("..."):
                scenarios = generate_scenarios(...)
                results = run_batch_test(...)
                vuln = find_vulnerable_nodes(...)
              st.session_state["results"] = results
              st.session_state["vuln"] = vuln

Step 8.4.2: Display results (if they exist in session state):

  RESULT 1: Big Resilience Score (3 metric columns)
  - Resilience Score (avg_fill/100)
  - Avg Fill Rate Under Stress
  - Avg Cost Increase %
  - Color-coded alert (success/warning/error)

  RESULT 2: Vulnerability Bar Chart
  - Horizontal bar chart of top 10 most vulnerable nodes
  - Color: crimson
  - X-axis: Fill Rate Drop (%)

  RESULT 3: Baseline vs Stress Comparison
  - Grouped bar chart comparing Fill Rate and Cost Index

  RESULT 4: Fill Rate Distribution Histogram
  - Histogram of fill rates across all Monte Carlo runs
  - Title shows number of runs

  RESULT 5: Per-Market Impact Table
  - DataFrame showing each market's baseline vs stressed fill rate

  RESULT 6: Mitigation Suggestions
  - Based on top vulnerable nodes, show context-aware suggestions
```

### Module 8.5: Season Effects

**Implementation Instructions:**
```
Step 8.5.1: Before running stress test, apply season modifier:
            if season == "Winter (Peak Demand)":
              # Increase all market demands by 25%
              modify demand_mt_day in nodes_df
            elif season == "Monsoon (Port Risk)":
              # Reduce port capacities by 20%
              modify port node capacities

Step 8.5.2: Reset after test completes
```

### Module 8.6: Git Commit

```
git add app.py
git commit -m "feat: complete Streamlit dashboard with map and results"
git push origin main
```

**✅ Checkpoint:** Dashboard loads, map renders, stress test runs, all 6 result components display.

---

## PHASE 9: Integration & Testing

**Duration:** 1-2 hours
**Goal:** End-to-end testing, bug fixes, edge cases

### Module 9.1: Test All Disruption Types

```
Step 9.1.1: Run app, test "Red Sea / Shipping Crisis"
            Expected: Sea costs high, port throughput down
Step 9.1.2: Test "Refinery Shutdown"
            Expected: Domestic production down, imports unaffected
Step 9.1.3: Test "Cyclone / Natural Disaster"
            Expected: Regional impact only
Step 9.1.4: Test "Price Shock"
            Expected: Costs spike, fill rate slight drop
Step 9.1.5: Test "Multi-Factor"
            Expected: Worst overall impact
```

### Module 9.2: Test Severity Slider

```
Step 9.2.1: Run with severity=0.1 → Minimal impact
Step 9.2.2: Run with severity=0.5 → Moderate impact
Step 9.2.3: Run with severity=1.0 → Severe impact
Step 9.2.4: Verify: higher severity → lower fill rate
```

### Module 9.3: Test Edge Cases

```
Step 9.3.1: Run with n_scenarios=50 → Should work
Step 9.3.2: Run with n_scenarios=200 → Should work (< 60 sec)
Step 9.3.3: Run multiple times → Results should vary slightly (Monte Carlo)
```

### Module 9.4: Write Unit Tests

**File:** `tests/test_digital_twin.py`
```
Step 9.4.1: Test load_network loads correct number of nodes/edges
Step 9.4.2: Test baseline fill_rate is between 80-100%
Step 9.4.3: Test disruption reduces fill_rate
Step 9.4.4: Test reset restores original values
```

### Module 9.5: Git Commit

```
git add tests/
git commit -m "test: add unit tests for digital twin and stress tester"
git push origin main
```

**✅ Checkpoint:** All 5 disruption types work. Severity slider works. No crashes.

---

## PHASE 10: Deployment & Documentation

**Duration:** 1-2 hours
**Goal:** Deploy to Streamlit Cloud, write README, final polish

### Module 10.1: Prepare for Deployment

```
Step 10.1.1: Verify requirements.txt has ALL dependencies
Step 10.1.2: Add to requirements.txt if deploying:
             --extra-index-url https://download.pytorch.org/whl/cpu
             torch
             (This installs CPU-only PyTorch — much smaller)

Step 10.1.3: Create .streamlit/config.toml:
             [theme]
             primaryColor = "#FF4B4B"
             backgroundColor = "#FFFFFF"
             font = "sans serif"

Step 10.1.4: Make sure models/saved_generator.pt is in the repo
             (remove it from .gitignore if you added it)

Step 10.1.5: Verify app.py is at repo root (not inside any folder)
```

### Module 10.2: Push Final Code to GitHub

```
Step 10.2.1: git add .
Step 10.2.2: git status (review all files)
Step 10.2.3: git commit -m "release: v1.0 - complete stress tester"
Step 10.2.4: git push origin main
```

### Module 10.3: Deploy to Streamlit Cloud

```
Step 10.3.1: Go to share.streamlit.io
Step 10.3.2: Click "New app"
Step 10.3.3: Select your GitHub repo
Step 10.3.4: Branch: main
Step 10.3.5: Main file: app.py
Step 10.3.6: Click "Deploy"
Step 10.3.7: Wait 5-10 minutes for build
Step 10.3.8: Test the deployed URL
```

### Module 10.4: Write README.md

```
Step 10.4.1: Open README.md
Step 10.4.2: Write all sections (see Section 5 for complete template)
Step 10.4.3: Take screenshots of working app
Step 10.4.4: Add screenshots to README (upload to GitHub)
```

### Module 10.5: Final Git Commit

```
git add README.md .streamlit/
git commit -m "docs: complete README with architecture and usage"
git push origin main
```

**✅ FINAL CHECKPOINT:** App deployed, README complete, all code on GitHub.

---

# 5. FILE-BY-FILE IMPLEMENTATION GUIDE

Below is the **exact content** for each file. When I get your approval, I'll implement each file fully.

## File List:

| # | File | Lines (approx) | Purpose |
|---|------|-----------------|---------|
| 1 | `requirements.txt` | 12 | Dependencies |
| 2 | `.gitignore` | 20 | Git ignore rules |
| 3 | `data/nodes.csv` | 21 | Network nodes |
| 4 | `data/edges.csv` | 27 | Network edges |
| 5 | `engine/__init__.py` | 3 | Package init |
| 6 | `engine/digital_twin.py` | ~180 | Core network model |
| 7 | `engine/stress_tester.py` | ~120 | Batch testing |
| 8 | `gan/__init__.py` | 3 | Package init |
| 9 | `gan/data_collector.py` | ~250 | Real data collection |
| 10 | `gan/model.py` | ~90 | GAN architecture |
| 11 | `gan/train.py` | ~150 | Training loop |
| 12 | `gan/generate.py` | ~200 | Scenario generation |
| 13 | `app.py` | ~350 | Streamlit dashboard |
| 14 | `tests/test_digital_twin.py` | ~60 | Unit tests |
| 15 | `README.md` | ~150 | Documentation |

**Total: ~1,600 lines of code across 15 files**

---

# 6. TESTING CHECKPOINTS SUMMARY

```
AFTER PHASE 2: ✓ Data validates (supply > demand, no duplicates)
AFTER PHASE 3: ✓ Baseline fill rate 85-95%
               ✓ Disruptions cause measurable drops
               ✓ Reset works
AFTER PHASE 4: ✓ Batch test runs 100 scenarios < 30 sec
               ✓ Vulnerability ranking: ports > bottling > refineries
AFTER PHASE 5: ✓ training_scenarios.csv has 5000+ rows
AFTER PHASE 6: ✓ GAN trains without NaN
               ✓ Losses converge
AFTER PHASE 7: ✓ Generated scenarios are diverse (not identical)
               ✓ Full pipeline works end-to-end
AFTER PHASE 8: ✓ Dashboard loads, map renders
               ✓ All 5 disruption types work
               ✓ All 6 result components display
AFTER PHASE 9: ✓ Severity slider scales impact
               ✓ Unit tests pass
AFTER PHASE 10: ✓ Deployed URL works
                ✓ README is complete
```

---

# 7. GITHUB WORKFLOW

```
BRANCHING STRATEGY (Simplified):
═══════════════════════════════

Work on main branch directly (3-day project, simple workflow).

COMMIT AFTER EACH PHASE:
  Phase 1: "feat: initial project structure"
  Phase 2: "feat: validated supply chain data"
  Phase 3: "feat: digital twin with max-flow"
  Phase 4: "feat: stress tester with vulnerability analysis"
  Phase 5: "feat: GAN training data from real sources"
  Phase 6: "feat: GAN model and training"
  Phase 7: "feat: scenario generation with validation"
  Phase 8: "feat: Streamlit dashboard"
  Phase 9: "test: unit tests and integration"
  Phase 10: "docs: README and deployment"

COMMIT CONVENTION:
  feat: new feature
  fix: bug fix
  docs: documentation
  test: tests
  refactor: code restructuring
```

---

# 8. TROUBLESHOOTING GUIDE

```
PROBLEM                          SOLUTION
═══════════════════════════════════════════════════════════
max_flow_min_cost errors         Cast capacities to int()
                                 OR use nx.maximum_flow()

Baseline fill rate < 80%        Edge capacities too low.
                                 Increase bottleneck edges.

Baseline fill rate = 100%       Normal! It means supply
                                 chain has enough slack.

GAN outputs all identical       Mode collapse. Reduce D
(mode collapse)                  learning rate to 0.0001.
                                 Or train D every 3rd batch.

GAN losses are NaN              Gradient explosion. Reduce
                                 learning rate to 0.00005.
                                 Increase gradient clipping.

Streamlit Cloud fails           PyTorch too large. Use
(memory/size)                    CPU-only: torch --index-url
                                 https://download.pytorch.org
                                 /whl/cpu

yfinance download fails         API rate limit or network.
                                 Wrap in try/except. Use
                                 cached data as fallback.

Map shows Saudi Arabia too big  Use scope="asia" with
                                 center=dict(lat=22, lon=78)
                                 and projection_scale=4

Import errors                   Check __init__.py files exist.
                                 Use: python -m module.name
                                 not: python module/name.py

"No module named engine"        Run from project root:
                                 cd supply-chain-stress-tester
                                 THEN: streamlit run app.py
```

---

# 9. README.md TEMPLATE

```markdown
# 🔬 India LPG Supply Chain Resilience Stress Tester

> GAN-powered disruption scenario generation and Monte Carlo
> stress testing for India's LPG supply chain network.

## 🎯 What This Does

This tool models India's LPG supply chain as a network graph 
(20 nodes, 26 edges) and uses a Conditional GAN trained on 
**real historical data** (PPAC reports, commodity prices) to 
generate realistic disruption scenarios. It then runs 100 
Monte Carlo simulations to calculate a **Resilience Score** 
and identify the most **vulnerable nodes** in the network.

## 🏗️ Architecture

[ASCII diagram of pipeline]

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation
git clone https://github.com/YOUR_USERNAME/supply-chain-stress-tester.git
cd supply-chain-stress-tester
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

### Train the GAN (first time only)
python -m gan.data_collector
python -m gan.train

### Run the Dashboard
streamlit run app.py

## 📊 Features
- 5 disruption types (Red Sea, Cyclone, Refinery, Price, Multi)
- Severity slider (0.1 - 1.0)
- Season effects (Normal, Winter, Monsoon)
- 100 Monte Carlo simulations
- Interactive India map with vulnerability coloring
- Resilience score with mitigation suggestions

## 🔧 Tech Stack
- **Network Model:** NetworkX (max-flow min-cost)
- **GAN:** PyTorch (Conditional GAN)
- **Dashboard:** Streamlit + Plotly
- **Data:** PPAC, yfinance, EM-DAT

## 📁 Project Structure
[folder tree]

## 📈 Sample Results
[screenshots]

## 🎓 Academic Context
Built as a supply chain resilience analysis project
demonstrating GAN-based scenario generation for
stress testing critical infrastructure networks.
```

---

# DECISION POINT

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  This is the complete plan covering:                        │
│                                                             │
│  ✅ 10 Phases with clear modules and substeps               │
│  ✅ 15 files with exact purposes and line counts             │
│  ✅ VSCode-specific instructions (terminal commands)         │
│  ✅ GitHub workflow (commits after each phase)               │
│  ✅ Testing checkpoints after each phase                     │
│  ✅ Troubleshooting guide                                    │
│  ✅ README template                                          │
│  ✅ Deployment instructions                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```