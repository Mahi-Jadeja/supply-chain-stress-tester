# Global Gas Pipeline Network Stress Tester — Complete Execution Plan

---

## 1. Problem Statement

Natural gas pipeline networks are the backbone of global energy infrastructure, transporting fuel across thousands of kilometers to power homes, industries, and electricity generation. These networks face constant threats from multiple directions — hurricanes devastating Gulf Coast infrastructure (Hurricane Katrina 2005, Hurricane Harvey 2017), cyberattacks shutting down critical pipelines (Colonial Pipeline ransomware 2021), geopolitical conflicts cutting supply routes (Russia-Ukraine gas disputes 2009, 2014, 2022), infrastructure sabotage destroying undersea pipelines (Nord Stream explosion 2022), and extreme weather freezing pipeline systems (Texas Winter Storm Uri 2021).

Despite the critical importance of these networks, existing risk assessment tools suffer from three fundamental limitations. First, they typically analyze pipeline failures in isolation — removing one pipeline at a time and measuring impact — which fails to capture the reality that disruptions are correlated. A hurricane does not knock out a single random pipeline; it simultaneously disables every pipeline in a geographic cluster. Second, most analyses rely on synthetic or simplified network models rather than real infrastructure data with actual capacities, routes, and ownership information. Third, there is no accessible tool that combines generative AI with real pipeline data to produce novel but realistic disruption scenarios that go beyond historical events.

This project addresses these gaps by building a stress testing framework that uses the GEM-GGIT Global Gas Pipelines Dataset (real coordinates, real capacities, real ownership data) combined with a Conditional Generative Adversarial Network trained on patterns from 15+ documented real-world disruption events. The system identifies which hubs are single points of failure, which companies own dangerously concentrated shares of critical infrastructure, and how the network fragments under various categories of stress.

---

## 2. Objectives

**Objective 1:** Convert the GEM-GGIT GeoJSON dataset of real global gas pipelines into a graph-based network model where nodes represent infrastructure hubs, junctions, and terminals extracted through geographic clustering of pipeline endpoints, and edges represent pipeline segments carrying real capacity and length data.

**Objective 2:** Build a digital twin simulation engine that loads this graph, calculates network health metrics (connectivity, total capacity, edge connectivity, betweenness centrality), applies disruption scenarios by removing nodes or reducing edge capacities, measures the resulting degradation, and resets to baseline for the next test.

**Objective 3:** Develop a Conditional GAN trained on disruption patterns derived from 15+ documented real-world events and Henry Hub natural gas price data, capable of generating novel but realistic multi-point failure scenarios across five disruption archetypes (natural disaster, sabotage, geopolitical sanctions, demand surge, multi-factor combinations).

**Objective 4:** Create an interactive Streamlit dashboard that displays actual pipeline routes on a geographic map, allows users to select disruption types and severities, runs stress tests, visualizes vulnerability rankings, and shows corporate ownership concentration risk.

**Objective 5:** Demonstrate that GAN-generated correlated failure scenarios reveal infrastructure vulnerabilities that simple one-at-a-time node removal analysis would miss.

---

## 3. Dataset Details

```
Source:  GEM-GGIT Global Gas Pipelines Dataset (2025)
Format: GeoJSON FeatureCollection
File:   GEM-GGIT-Gas-Pipelines-2025-11.geojson

Fields per pipeline feature:
─────────────────────────────
PipelineName          → name of the pipeline system
SegmentName           → specific segment within the system
Status                → operating, proposed, shelved, cancelled, etc.
CountriesOrAreas      → which countries the pipeline passes through
Owner                 → operating company
CapacityBcm/y         → annual capacity in billion cubic meters
LengthMergedKm        → total length in kilometers
Diameter              → pipe diameter in inches
CostUSD               → construction cost
StartLocation         → starting city/area name
EndLocation           → ending city/area name
StartState/Province   → starting state
EndState/Province     → ending state
StartCountryOrArea    → starting country
EndCountryOrArea      → ending country
Geometry              → MultiLineString with actual route coordinates
                        (arrays of [longitude, latitude] pairs)

Regional Focus: United States only
Filter: CountriesOrAreas contains "United States"
        AND Status == "operating"
Expected result: 100-300 pipelines with capacity data
```

---

## 4. Execution Phases

---

### PHASE 1: PROJECT SETUP

**What you are doing:** Creating the folder structure, installing libraries, and verifying everything works before writing any real code.

**Step 1.1 — Create the folder structure**

Create these folders and empty files exactly as shown:

```
cd /Users/mj/Desktop/GANS

# Add all new files
git add .

# Commit
git commit -m "Add project structure: directories, empty modules, requirements, .gitignore"

# Push
git push origin main

The `__init__.py` files must exist (even if empty) so Python treats those folders as importable packages. If you forget them, you will get `ModuleNotFoundError` later.

**Step 1.2 — Create requirements.txt**

```
torch
numpy
pandas
networkx
plotly
streamlit
scikit-learn
yfinance
geopy
```

**Step 1.3 — Install dependencies**

```bash
cd gas-pipeline-stress-tester
pip install -r requirements.txt
```

**What can go wrong here and how to fix it:**

```
PROBLEM: "pip install torch" downloads a 2GB file and takes forever
         or fails with timeout
FIX:     Install CPU-only version instead:
         pip install torch --index-url https://download.pytorch.org/whl/cpu
         This is 200MB instead of 2GB and works fine for this project
         since we are training a tiny GAN, not a large model.

PROBLEM: "pip install geopy" fails with permission error
FIX:     Use: pip install --user geopy
         Or use a virtual environment:
         python -m venv venv
         source venv/bin/activate  (Linux/Mac)
         venv\Scripts\activate     (Windows)
         pip install -r requirements.txt

PROBLEM: "No module named torch" even after installing
FIX:     You might have multiple Python versions.
         Check: python --version
         Use: python -m pip install torch
         Or: python3 -m pip install torch

PROBLEM: yfinance install fails
FIX:     pip install yfinance --upgrade
         If still fails: pip install yfinance==0.2.31
         (specific version that is known to work)
```

**Step 1.4 — Verify installations**

Create a file called `test_imports.py`:

```python
print("Testing imports...")
import torch
print(f"  torch {torch.__version__} OK")
import numpy as np
print(f"  numpy {np.__version__} OK")
import pandas as pd
print(f"  pandas {pd.__version__} OK")
import networkx as nx
print(f"  networkx {nx.__version__} OK")
import plotly
print(f"  plotly {plotly.__version__} OK")
import streamlit
print(f"  streamlit {streamlit.__version__} OK")
import sklearn
print(f"  sklearn {sklearn.__version__} OK")
import yfinance
print(f"  yfinance {yfinance.__version__} OK")
import geopy
print(f"  geopy OK")
print("\nAll imports successful!")
```

Run it: `python test_imports.py`

Every line should print OK. If any fails, fix that specific library before continuing.

**Step 1.5 — Place the GeoJSON file**

Copy your `GEM-GGIT-Gas-Pipelines-2025-11.geojson` file into `data/raw/`.

Verify you can load it:

```python
import json
with open("data/raw/GEM-GGIT-Gas-Pipelines-2025-11.geojson") as f:
    data = json.load(f)
print(f"Type: {data['type']}")
print(f"Number of features: {len(data['features'])}")
print(f"First feature properties keys: {list(data['features'][0]['properties'].keys())}")
print(f"First feature geometry type: {data['features'][0]['geometry']['type']}")
```

```
PROBLEM: json.load() takes very long or runs out of memory
WHY:     The GeoJSON file might be 50-200MB with thousands of
         pipelines worldwide
FIX:     This is normal. It will take 5-30 seconds to load.
         If it crashes with MemoryError, you need at least 4GB RAM.
         Alternative: Use ijson library for streaming JSON parsing:
         pip install ijson
         But try regular json.load() first — it usually works.

PROBLEM: FileNotFoundError
FIX:     Check exact filename. It might have a different name
         or different extension. Use:
         import os
         print(os.listdir("data/raw/"))
         to see what files are actually there.

PROBLEM: UnicodeDecodeError
FIX:     Open with encoding specified:
         with open(filepath, encoding="utf-8") as f:

PROBLEM: json.JSONDecodeError
FIX:     File might be corrupted or not actually JSON.
         Open it in a text editor and check the first few lines.
         Should start with {"type": "FeatureCollection"
```

**Step 1.6 — Explore the data structure**

This is critical. Before writing any processing code, understand exactly what your data looks like:

```python
import json

with open("data/raw/GEM-GGIT-Gas-Pipelines-2025-11.geojson") as f:
    data = json.load(f)

features = data["features"]

# Look at the first feature in detail
f0 = features[0]
print("=== PROPERTIES ===")
for key, value in f0["properties"].items():
    print(f"  {key}: {value} (type: {type(value).__name__})")

print("\n=== GEOMETRY ===")
geom = f0["geometry"]
print(f"  Type: {geom['type']}")
print(f"  Number of line segments: {len(geom['coordinates'])}")
print(f"  First segment, first 3 points: {geom['coordinates'][0][:3]}")
print(f"  First segment, last 3 points: {geom['coordinates'][0][-3:]}")

# Check what countries are present
countries = set()
for f in features:
    c = f["properties"].get("CountriesOrAreas", "")
    if c:
        countries.add(c)
print(f"\nUnique countries/areas: {len(countries)}")
for c in sorted(countries)[:20]:
    print(f"  {c}")

# Check what statuses are present
statuses = {}
for f in features:
    s = f["properties"].get("Status", "unknown")
    statuses[s] = statuses.get(s, 0) + 1
print(f"\nStatus distribution:")
for s, count in sorted(statuses.items(), key=lambda x: -x[1]):
    print(f"  {s}: {count}")

# Check US operating pipelines with capacity
us_operating = []
for f in features:
    props = f["properties"]
    countries_field = props.get("CountriesOrAreas", "")
    status = props.get("Status", "")
    capacity = props.get("CapacityBcm/y", "")
    if ("United States" in str(countries_field) and
        status == "operating"):
        us_operating.append(f)

print(f"\nUS operating pipelines: {len(us_operating)}")

us_with_capacity = [f for f in us_operating
                    if f["properties"].get("CapacityBcm/y", "") != ""]
print(f"US operating with capacity data: {len(us_with_capacity)}")
```

**Why this matters:**

```
YOU MUST RUN THIS EXPLORATION STEP.
The exact field names, data types, and values in YOUR file
might differ slightly from what is documented.

THINGS THAT MIGHT SURPRISE YOU:

1. The field might be "CapacityBcm/y" or "CapacityBcm_y" or
   "Capacity (Bcm/y)" — you need to check the exact key name.

2. CountriesOrAreas might contain "United States" or "USA" or
   "US" — you need to check which string to filter on.

3. Status might be "Operating" (capital O) not "operating" —
   string matching is case-sensitive.

4. Some fields might be None instead of "" (empty string).

5. Geometry coordinates might be nested differently than expected.
   MultiLineString = list of list of [lon, lat] pairs
   But some might be deeper nested.

WRITE DOWN THE EXACT FIELD NAMES YOU SEE.
You will use them in the next phase.
```

**Done when:**
- All libraries install and import
- GeoJSON file loads successfully
- You know the exact field names for: pipeline name, status, country, capacity, length, owner, diameter, cost, start/end locations, geometry
- You know how many US operating pipelines exist
- You know how many have capacity data

---

### PHASE 2: DATA PROCESSING — GeoJSON to Graph

**What you are doing:** Converting the raw GeoJSON file into two clean CSV files (nodes.csv and edges.csv) that represent the pipeline network as a graph.

**File:** `engine/data_processor.py`

---

#### Phase 2A — Loading and Filtering

```python
import json
import pandas as pd
import numpy as np
from geopy.distance import geodesic

def load_geojson(filepath):
    """Load GeoJSON file and return the features list."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data["features"]
```

```
WHAT CAN GO WRONG:
  Nothing new here — you already tested this in Phase 1.
  But wrap it in a function so the rest of your code calls this.
```

```python
def filter_pipelines(features, country="United States",
                     status="operating"):
    """
    Filter to keep only pipelines matching country and status.
    Also require that the pipeline has geometry data.
    """
    filtered = []
    skipped_no_country = 0
    skipped_no_status = 0
    skipped_no_geometry = 0

    for f in features:
        props = f["properties"]

        # Check country
        countries_field = str(props.get("CountriesOrAreas", ""))
        if country not in countries_field:
            skipped_no_country += 1
            continue

        # Check status (case-insensitive comparison)
        pipeline_status = str(props.get("Status", "")).lower()
        if pipeline_status != status.lower():
            skipped_no_status += 1
            continue

        # Check geometry exists and is not empty
        geom = f.get("geometry")
        if (geom is None or
            geom.get("coordinates") is None or
            len(geom["coordinates"]) == 0):
            skipped_no_geometry += 1
            continue

        filtered.append(f)

    print(f"Filtering results:")
    print(f"  Total features: {len(features)}")
    print(f"  Skipped (wrong country): {skipped_no_country}")
    print(f"  Skipped (wrong status): {skipped_no_status}")
    print(f"  Skipped (no geometry): {skipped_no_geometry}")
    print(f"  Kept: {len(filtered)}")

    return filtered
```

```
WHAT CAN GO WRONG:

PROBLEM: filter_pipelines returns 0 results
CAUSE 1: Country name mismatch
  → Your file might use "USA" instead of "United States"
  → FIX: Print the first 5 values of CountriesOrAreas
    to see the exact format:
    for f in features[:20]:
        print(f["properties"].get("CountriesOrAreas"))

CAUSE 2: Status field is capitalized differently
  → "Operating" vs "operating" vs "OPERATING"
  → FIX: Already handled with .lower() comparison above

CAUSE 3: CountriesOrAreas contains multiple countries
  → Example: "United States; Mexico" for cross-border pipelines
  → FIX: Using "in" operator already handles this
    ("United States" in "United States; Mexico" is True)

PROBLEM: filter_pipelines returns too few results (under 30)
CAUSE:   Most US pipelines might not have "operating" status
         or might have a different status label
FIX:     Remove the status filter temporarily:
         filtered = filter_pipelines(features, "United States", None)
         See how many you get. Then check what statuses exist:
         statuses = set(f["properties"]["Status"] for f in filtered)
         print(statuses)

PROBLEM: filter_pipelines returns too many results (over 500)
CAUSE:   This is actually fine for data processing.
         The clustering step will reduce the node count.
         Only worry if the code becomes too slow later.
```

---

#### Phase 2B — Extracting Pipeline Endpoints

This function gets the start and end coordinates from each pipeline's geometry, plus all the metadata.

```python
def extract_endpoints(features):
    """
    For each pipeline feature, extract:
    - Start coordinate (first point of first line segment)
    - End coordinate (last point of last line segment)
    - All metadata (capacity, length, owner, etc.)
    """
    endpoints = []
    skipped = 0

    for f in features:
        props = f["properties"]
        geom = f["geometry"]
        coords = geom["coordinates"]

        # CRITICAL: GeoJSON MultiLineString structure is:
        # coordinates = [
        #   [ [lon1,lat1], [lon2,lat2], ... ],  ← segment 1
        #   [ [lon1,lat1], [lon2,lat2], ... ],  ← segment 2
        #   ...
        # ]
        #
        # Each segment is a list of [longitude, latitude] pairs.
        # NOTE: GeoJSON uses [LONGITUDE, LATITUDE] order!
        # This is the OPPOSITE of what most people expect.
        # Getting this wrong puts your pipelines in the ocean.

        try:
            first_segment = coords[0]
            last_segment = coords[-1]

            # Start = first point of first segment
            start_lon = first_segment[0][0]
            start_lat = first_segment[0][1]

            # End = last point of last segment
            end_lon = last_segment[-1][0]
            end_lat = last_segment[-1][1]

            # SANITY CHECK: coordinates should be in valid range
            # Latitude: -90 to 90
            # Longitude: -180 to 180
            # For US: lat should be roughly 24-50, lon -125 to -65
            if not (-90 <= start_lat <= 90 and -180 <= start_lon <= 180):
                print(f"  WARNING: Invalid start coords for "
                      f"{props.get('PipelineName')}: "
                      f"({start_lat}, {start_lon})")
                skipped += 1
                continue

            if not (-90 <= end_lat <= 90 and -180 <= end_lon <= 180):
                print(f"  WARNING: Invalid end coords for "
                      f"{props.get('PipelineName')}: "
                      f"({end_lat}, {end_lon})")
                skipped += 1
                continue

        except (IndexError, TypeError) as e:
            print(f"  WARNING: Could not extract coords for "
                  f"{props.get('PipelineName')}: {e}")
            skipped += 1
            continue

        # Parse capacity — might be string, float, int, or empty
        capacity_raw = props.get("CapacityBcm/y", "")
        try:
            capacity_bcm_y = float(capacity_raw) if capacity_raw else 0.0
        except (ValueError, TypeError):
            capacity_bcm_y = 0.0

        # Parse length
        length_raw = props.get("LengthMergedKm", "")
        # MIGHT ALSO BE "LengthKnownKm" or "LengthEstimateKm"
        # Check your data exploration output from Phase 1
        if not length_raw:
            length_raw = props.get("LengthKnownKm", "")
        if not length_raw:
            length_raw = props.get("LengthEstimateKm", "")
        try:
            length_km = float(length_raw) if length_raw else 0.0
        except (ValueError, TypeError):
            length_km = 0.0

        # Parse cost
        cost_raw = props.get("CostUSD", "")
        try:
            cost_usd = float(cost_raw) if cost_raw else 0.0
        except (ValueError, TypeError):
            cost_usd = 0.0

        # Parse diameter
        diameter_raw = props.get("Diameter", "")
        try:
            diameter = float(diameter_raw) if diameter_raw else 0.0
        except (ValueError, TypeError):
            diameter = 0.0

        endpoints.append({
            "pipeline_name": props.get("PipelineName", "Unknown"),
            "segment_name": props.get("SegmentName", ""),
            "start_lat": start_lat,
            "start_lon": start_lon,
            "end_lat": end_lat,
            "end_lon": end_lon,
            "capacity_bcm_y": capacity_bcm_y,
            "length_km": length_km,
            "owner": props.get("Owner", "Unknown"),
            "status": props.get("Status", ""),
            "diameter": diameter,
            "cost_usd": cost_usd,
            "start_state": props.get("StartState/Province", ""),
            "end_state": props.get("EndState/Province", ""),
            "start_country": props.get("StartCountryOrArea", ""),
            "end_country": props.get("EndCountryOrArea", ""),
            # Store geometry as JSON string for later map drawing
            "geometry": json.dumps(coords),
        })

    print(f"Extracted {len(endpoints)} endpoints, skipped {skipped}")
    return endpoints
```

```
WHAT CAN GO WRONG:

PROBLEM: All coordinates are (0, 0) or clearly wrong
CAUSE:   You mixed up longitude and latitude
         GeoJSON is [longitude, latitude] NOT [latitude, longitude]
FIX:     If you see lat values like -95 or -120 (these are US longitudes,
         not latitudes), you have them swapped.
         Swap: start_lat = first_segment[0][0]  ← WRONG
               start_lon = first_segment[0][1]  ← WRONG
         To:   start_lon = first_segment[0][0]  ← CORRECT
               start_lat = first_segment[0][1]  ← CORRECT

PROBLEM: IndexError when accessing coords[0][0][0]
CAUSE:   Some geometries might be LineString instead of MultiLineString
         LineString coordinates = [[lon,lat], [lon,lat], ...]
         MultiLineString coordinates = [[[lon,lat], ...], [[lon,lat], ...]]
FIX:     Check geometry type and handle both:
         if geom["type"] == "MultiLineString":
             first_segment = coords[0]
             start_lon = first_segment[0][0]
         elif geom["type"] == "LineString":
             start_lon = coords[0][0]
         This is a VERY COMMON bug. Add this check.

PROBLEM: Some coordinates have 3 values [lon, lat, elevation]
CAUSE:   3D coordinates are valid GeoJSON
FIX:     Just take the first two values:
         start_lon = first_segment[0][0]
         start_lat = first_segment[0][1]
         (ignore [2] which is elevation — already handled above)

PROBLEM: capacity_bcm_y is 0 for most pipelines
CAUSE:   The field name might be different in your file
FIX:     Check your Phase 1 exploration output from Phase 1.
         Try "Capacity (Bcm/y)" or "CapacityBcm_y" etc.
         Also try the MMcf/d field and convert:
         1 MMcf/d ≈ 0.01034 BCM/y
         capacity_bcm_y = float(props["CapacityMMcf/d"]) * 0.01034

PROBLEM: Pipeline names contain special characters or are very long
CAUSE:   Normal for real data
FIX:     Not a problem for processing. Just be aware for display.
```

**VERIFICATION STEP — Run this before continuing:**

```python
# Test the extraction
features = load_geojson("data/raw/GEM-GGIT-Gas-Pipelines-2025-11.geojson")
filtered = filter_pipelines(features, "United States", "operating")
endpoints = extract_endpoints(filtered)

# Quick sanity check
print(f"\nSanity check on first 5 endpoints:")
for ep in endpoints[:5]:
    print(f"  {ep['pipeline_name']}: "
          f"({ep['start_lat']:.2f}, {ep['start_lon']:.2f}) → "
          f"({ep['end_lat']:.2f}, {ep['end_lon']:.2f}), "
          f"cap={ep['capacity_bcm_y']}, len={ep['length_km']}")

# Check coordinate ranges (should be within US bounds)
lats = [ep["start_lat"] for ep in endpoints] + [ep["end_lat"] for ep in endpoints]
lons = [ep["start_lon"] for ep in endpoints] + [ep["end_lon"] for ep in endpoints]
print(f"\nLatitude range: {min(lats):.2f} to {max(lats):.2f}")
print(f"  (expected: roughly 24 to 50 for US)")
print(f"Longitude range: {min(lons):.2f} to {max(lons):.2f}")
print(f"  (expected: roughly -125 to -65 for US)")

# Check how many have capacity data
with_cap = [ep for ep in endpoints if ep["capacity_bcm_y"] > 0]
print(f"\nPipelines with capacity data: {len(with_cap)} / {len(endpoints)}")
```

```
IF LATITUDE RANGE IS -125 TO -65:
  You swapped lat and lon. Go back and fix.

IF LATITUDE RANGE IS 24 TO 50 AND LONGITUDE IS -125 TO -65:
  Correct! Continue.

IF MOST PIPELINES HAVE ZERO CAPACITY:
  That is okay. We will assign default capacity values.
  As long as SOME pipelines have real capacity, the model works.
  Pipelines without capacity get a default of 0.01 BCM/y.
```

---

#### Phase 2C — Clustering Endpoints into Nodes

This is the most conceptually complex part. Multiple pipelines might connect to the same physical location (like Henry Hub in Louisiana) but have slightly different coordinates (5-15 km apart because they connect at different sides of the facility). We need to merge these into single nodes.

```python
def cluster_endpoints_into_nodes(endpoints, radius_km=15):
    """
    Group nearby pipeline endpoints into single nodes.

    Algorithm:
    1. Collect ALL start and end points from all pipelines
    2. For each point, check if it is within radius_km of
       any existing node center
    3. If yes, assign it to that node
    4. If no, create a new node at that location
    5. Classify nodes by connection count:
       3+ connections = hub
       2 connections = junction
       1 connection = terminal

    Parameters:
    -----------
    endpoints : list of dicts from extract_endpoints()
    radius_km : float, clustering radius in kilometers
                15 km is a good default for US pipelines

    Returns:
    --------
    nodes : list of dicts with id, lat, lon, type, label, members
    point_to_node : dict mapping point index to node id
    """
    # Step 1: Collect ALL points (each pipeline contributes 2)
    all_points = []
    for ep in endpoints:
        all_points.append({
            "lat": ep["start_lat"],
            "lon": ep["start_lon"],
            "label": ep["start_state"] if ep["start_state"] else "Unknown",
            "pipeline": ep["pipeline_name"],
            "side": "start"
        })
        all_points.append({
            "lat": ep["end_lat"],
            "lon": ep["end_lon"],
            "label": ep["end_state"] if ep["end_state"] else "Unknown",
            "pipeline": ep["pipeline_name"],
            "side": "end"
        })

    print(f"Total points to cluster: {len(all_points)}")

    # Step 2: Greedy clustering
    nodes = []
    point_to_node = {}

    for i, point in enumerate(all_points):
        # Print progress every 100 points
        if i % 100 == 0:
            print(f"  Clustering point {i}/{len(all_points)}...")

        matched_node = None
        point_coords = (point["lat"], point["lon"])

        for node in nodes:
            node_coords = (node["lat"], node["lon"])
            try:
                dist = geodesic(point_coords, node_coords).km
            except ValueError:
                # Invalid coordinates
                continue

            if dist < radius_km:
                matched_node = node
                break

        if matched_node:
            matched_node["members"].append(i)
            # Update label if current is better (non-empty)
            if point["label"] and point["label"] != "Unknown":
                if matched_node["label"] == "Unknown":
                    matched_node["label"] = point["label"]
            point_to_node[i] = matched_node["id"]
        else:
            new_id = f"N{len(nodes):04d}"
            new_node = {
                "id": new_id,
                "lat": point["lat"],
                "lon": point["lon"],
                "label": point["label"],
                "members": [i]
            }
            nodes.append(new_node)
            point_to_node[i] = new_id

    # Step 3: Classify nodes by connection count
    for node in nodes:
        n_conn = len(node["members"])
        if n_conn >= 3:
            node["type"] = "hub"
        elif n_conn == 2:
            node["type"] = "junction"
        else:
            node["type"] = "terminal"
        node["connections"] = n_conn

    # Summary
    hubs = [n for n in nodes if n["type"] == "hub"]
    junctions = [n for n in nodes if n["type"] == "junction"]
    terminals = [n for n in nodes if n["type"] == "terminal"]
    print(f"\nClustering complete:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Hubs (3+ connections): {len(hubs)}")
    print(f"  Junctions (2 connections): {len(junctions)}")
    print(f"  Terminals (1 connection): {len(terminals)}")

    return nodes, point_to_node
```

```
WHAT CAN GO WRONG:

PROBLEM: Clustering takes forever (more than 5 minutes)
CAUSE:   The algorithm is O(n × m) where n = number of points
         and m = number of nodes. With 500+ pipelines, that is
         1000+ points, each compared to potentially hundreds of nodes.
FIX 1:   This should still finish in 1-3 minutes. Be patient.
FIX 2:   If truly too slow, use scipy spatial KDTree instead:
         from scipy.spatial import KDTree
         This does nearest-neighbor search in O(log n) instead of O(n).
         But the simple loop should work for under 1000 points.

PROBLEM: Too many nodes (over 300)
CAUSE:   Clustering radius is too small
FIX:     Increase radius_km to 20, 25, or 30
         Run again and check counts.
         Target: 80-200 nodes.

PROBLEM: Too few nodes (under 30)
CAUSE:   Clustering radius is too large, merging distinct locations
FIX:     Decrease radius_km to 10 or 8

PROBLEM: Only 1-2 hub nodes, everything else is terminal
CAUSE 1: Most pipelines are point-to-point with no shared endpoints
         This is actually normal for some regions.
CAUSE 2: Radius too small — endpoints that should cluster are not
FIX:     Increase radius to 20-25 km. Also check if coordinates
         are correct (lat/lon swap would prevent clustering).

PROBLEM: geodesic() raises ValueError
CAUSE:   Invalid coordinates (NaN, None, or out of range)
FIX:     Already handled with try/except above.
         But also add a pre-filter:
         all_points = [p for p in all_points
                       if -90 <= p["lat"] <= 90
                       and -180 <= p["lon"] <= 180]

PROBLEM: All nodes cluster into ONE giant node
CAUSE:   All coordinates are identical (data parsing bug)
         or radius_km is set absurdly large (like 5000)
FIX:     Check that coordinates are actually different:
         for ep in endpoints[:10]:
             print(ep["start_lat"], ep["start_lon"],
                   ep["end_lat"], ep["end_lon"])
         If they are all the same, go back to Phase 2B
         and fix the coordinate extraction.
```

---

#### Phase 2D — Building Edges

```python
def build_edges(endpoints, point_to_node):
    """
    Convert pipelines into graph edges between clustered nodes.

    Each pipeline becomes an edge from its start node to its end node.
    If start and end cluster to the same node, skip (self-loop).
    """
    edges = []
    self_loops = 0
    missing_mapping = 0

    for i, ep in enumerate(endpoints):
        start_idx = i * 2        # Index in all_points list
        end_idx = i * 2 + 1

        # Check that mapping exists
        if start_idx not in point_to_node:
            missing_mapping += 1
            continue
        if end_idx not in point_to_node:
            missing_mapping += 1
            continue

        source_node = point_to_node[start_idx]
        target_node = point_to_node[end_idx]

        # Skip self-loops
        if source_node == target_node:
            self_loops += 1
            continue

        # Calculate daily capacity
        cap_bcm_y = ep["capacity_bcm_y"]
        if cap_bcm_y <= 0:
            cap_bcm_y = 0.01  # Default small capacity
        cap_bcm_day = cap_bcm_y / 365.0

        edges.append({
            "source": source_node,
            "target": target_node,
            "pipeline_name": ep["pipeline_name"],
            "capacity_bcm_y": cap_bcm_y,
            "capacity_bcm_day": cap_bcm_day,
            "length_km": ep["length_km"] if ep["length_km"] > 0 else 100.0,
            "owner": ep["owner"],
            "diameter": ep["diameter"],
            "cost_usd": ep["cost_usd"],
            "geometry": ep["geometry"],  # JSON string of coordinates
        })

    print(f"\nEdge building results:")
    print(f"  Total edges: {len(edges)}")
    print(f"  Self-loops skipped: {self_loops}")
    print(f"  Missing mappings: {missing_mapping}")

    # Check for duplicate edges (multiple pipelines between same nodes)
    edge_pairs = [(e["source"], e["target"]) for e in edges]
    unique_pairs = set(tuple(sorted(p)) for p in edge_pairs)
    print(f"  Unique node pairs: {len(unique_pairs)}")
    print(f"  Parallel pipelines: {len(edges) - len(unique_pairs)}")

    return edges
```

```
WHAT CAN GO WRONG:

PROBLEM: Most edges are self-loops (high self_loops count)
CAUSE:   The clustering radius is too large, causing start and end
         of the same pipeline to merge into one node.
         This happens when a pipeline is shorter than the radius.
         Example: 50 km pipeline with 30 km radius → both ends
         cluster together.
FIX:     Decrease radius_km. Try 10 or 8.
         OR: Only cluster if the pipeline length is greater
         than 2× the radius.

PROBLEM: Very few edges (under 30) despite having 100+ pipelines
CAUSE:   Same as above — too many self-loops
FIX:     Same as above — decrease radius

PROBLEM: missing_mapping count is high
CAUSE:   Some points were filtered out during clustering
         (invalid coordinates)
FIX:     Go back to extract_endpoints and ensure all coordinates
         pass the sanity check. Print the problematic ones.

PROBLEM: All edges have capacity 0.01 (default)
CAUSE:   No pipelines had capacity data, or wrong field name
FIX:     This is functional but not ideal.
         Check if the dataset has a different capacity field.
         Look for "CapacityMMcf/d" and convert:
         capacity_bcm_y = float(props["CapacityMMcf/d"]) * 365 * 0.00002832
         Or just proceed — the model still works with
         uniform capacities; it just measures connectivity
         rather than capacity.
```

---

#### Phase 2E — Saving Processed Data

```python
def save_processed_data(nodes, edges, output_dir="data/processed"):
    """Save nodes and edges as CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save nodes
    nodes_for_csv = []
    for n in nodes:
        nodes_for_csv.append({
            "id": n["id"],
            "lat": n["lat"],
            "lon": n["lon"],
            "type": n["type"],
            "label": n["label"],
            "connections": n["connections"]
        })
        # Do NOT save "members" list — it is internal data

    nodes_df = pd.DataFrame(nodes_for_csv)
    nodes_df.to_csv(f"{output_dir}/nodes.csv", index=False)
    print(f"Saved {len(nodes_df)} nodes to {output_dir}/nodes.csv")

    # Save edges
    edges_for_csv = []
    for e in edges:
        edges_for_csv.append({
            "source": e["source"],
            "target": e["target"],
            "pipeline_name": e["pipeline_name"],
            "capacity_bcm_y": e["capacity_bcm_y"],
            "capacity_bcm_day": e["capacity_bcm_day"],
            "length_km": e["length_km"],
            "owner": e["owner"],
            "diameter": e["diameter"],
            "cost_usd": e["cost_usd"],
            "geometry": e["geometry"],
        })

    edges_df = pd.DataFrame(edges_for_csv)
    edges_df.to_csv(f"{output_dir}/edges.csv", index=False)
    print(f"Saved {len(edges_df)} edges to {output_dir}/edges.csv")

    # VALIDATION
    print(f"\n=== VALIDATION ===")

    # Check every edge source and target exists in nodes
    node_ids = set(nodes_df["id"])
    bad_sources = edges_df[~edges_df["source"].isin(node_ids)]
    bad_targets = edges_df[~edges_df["target"].isin(node_ids)]
    if len(bad_sources) > 0:
        print(f"  ERROR: {len(bad_sources)} edges have unknown source nodes!")
    if len(bad_targets) > 0:
        print(f"  ERROR: {len(bad_targets)} edges have unknown target nodes!")
    if len(bad_sources) == 0 and len(bad_targets) == 0:
        print(f"  All edge endpoints exist in nodes: OK")

    # Check no self-loops
    self_loops = edges_df[edges_df["source"] == edges_df["target"]]
    if len(self_loops) > 0:
        print(f"  WARNING: {len(self_loops)} self-loops found!")
    else:
        print(f"  No self-loops: OK")

    # Check capacity range
    caps = edges_df["capacity_bcm_y"]
    print(f"  Capacity range: {caps.min():.4f} to {caps.max():.2f} BCM/y")

    return nodes_df, edges_df
```

---

#### Phase 2F — Main Processing Function

```python
def process_pipeline_data(geojson_path):
    """Main function that runs the complete processing pipeline."""
    print("=" * 60)
    print("GAS PIPELINE DATA PROCESSOR")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading GeoJSON...")
    features = load_geojson(geojson_path)
    print(f"  Loaded {len(features)} features")

    # Step 2: Filter
    print("\n[2/5] Filtering pipelines...")
    filtered = filter_pipelines(features, "United States", "operating")

    if len(filtered) == 0:
        print("ERROR: No pipelines matched the filter!")
        print("Try different country name or status.")
        print("Available countries (first 10):")
        countries = set()
        for f in features:
            c = str(f["properties"].get("CountriesOrAreas", ""))
            countries.add(c)
        for c in sorted(countries)[:10]:
            print(f"  {c}")
        return None, None

    # Step 3: Extract endpoints
    print("\n[3/5] Extracting endpoints...")
    endpoints = extract_endpoints(filtered)

    if len(endpoints) == 0:
        print("ERROR: No valid endpoints extracted!")
        print("Check coordinate extraction logic.")
        return None, None

    # Step 4: Cluster into nodes
    print("\n[4/5] Clustering endpoints into nodes...")
    nodes, point_to_node = cluster_endpoints_into_nodes(
        endpoints, radius_km=15
    )

    # Auto-adjust radius if needed
    if len(nodes) > 300:
        print(f"  Too many nodes ({len(nodes)}). "
              f"Re-clustering with radius=25km...")
        nodes, point_to_node = cluster_endpoints_into_nodes(
            endpoints, radius_km=25
        )

    if len(nodes) < 20:
        print(f"  Too few nodes ({len(nodes)}). "
              f"Re-clustering with radius=8km...")
        nodes, point_to_node = cluster_endpoints_into_nodes(
            endpoints, radius_km=8
        )

    # Step 5: Build edges
    print("\n[5/5] Building edges...")
    edges = build_edges(endpoints, point_to_node)

    if len(edges) == 0:
        print("ERROR: No edges created!")
        print("All pipelines became self-loops.")
        print("Try a smaller clustering radius.")
        return None, None

    # Save
    print("\n[SAVE] Saving processed data...")
    nodes_df, edges_df = save_processed_data(nodes, edges)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"  Nodes: {len(nodes_df)}")
    print(f"  Edges: {len(edges_df)}")
    print("=" * 60)

    return nodes_df, edges_df


# Allow running directly
if __name__ == "__main__":
    process_pipeline_data(
        "data/raw/GEM-GGIT-Gas-Pipelines-2025-11.geojson"
    )
```

**Run it:**
```bash
cd gas-pipeline-stress-tester
python -m engine.data_processor
```

```
WHAT CAN GO WRONG WHEN RUNNING:

PROBLEM: ModuleNotFoundError: No module named 'engine'
FIX:     Make sure you are running from the project ROOT directory
         (gas-pipeline-stress-tester/), not from inside engine/.
         Also make sure engine/__init__.py exists.
         Alternative: python engine/data_processor.py
         (but then you need to change the geojson path)

PROBLEM: MemoryError when loading GeoJSON
FIX:     The file is too large for your RAM.
         Option A: Filter the JSON file externally first.
         Open it in a text editor that handles large files
         (VS Code, Sublime Text) and manually trim.
         Option B: Process in chunks using ijson library.
         Option C: Use a machine with more RAM.

PROBLEM: It runs but output has weird numbers
FIX:     Run the verification steps printed during processing.
         Check the "VALIDATION" output.
         Open nodes.csv and edges.csv in Excel or a text editor
         and visually inspect.

EXPECTED OUTPUT:
  If everything works, you should see something like:
  Nodes: 80-200 (depends on dataset and radius)
  Edges: 50-300
  Hubs: 10-40
  All edge endpoints exist in nodes: OK
  No self-loops: OK
```

**Done when:**
- `data/processed/nodes.csv` exists with 50-200 rows
- `data/processed/edges.csv` exists with 30-300 rows
- Validation passes (no unknown nodes, no self-loops)
- Coordinate ranges are within US bounds
- At least some nodes are classified as "hub"

---

### PHASE 3: DIGITAL TWIN

**What you are doing:** Building a class that loads the processed graph, calculates network health metrics, and can apply/reset disruptions.

**File:** `engine/digital_twin.py`

```python
import networkx as nx
import pandas as pd
import numpy as np

class DigitalTwin:
    """
    Digital twin of a gas pipeline network.

    Loads the processed nodes/edges CSVs into a NetworkX graph
    and provides methods to:
    - Calculate network health KPIs
    - Apply disruption scenarios
    - Reset to baseline
    """

    def __init__(self):
        self.graph = None
        self.original_graph = None
        self.nodes_df = None
        self.edges_df = None
        self.baseline_kpis = None

    def load_network(self, nodes_csv, edges_csv):
        """
        Load the pipeline network from CSV files.
        Creates an undirected graph (gas flows both ways).
        """
        self.nodes_df = pd.read_csv(nodes_csv)
        self.edges_df = pd.read_csv(edges_csv)

        self.graph = nx.Graph()

        # Add nodes
        for _, row in self.nodes_df.iterrows():
            self.graph.add_node(
                row["id"],
                lat=row["lat"],
                lon=row["lon"],
                node_type=row["type"],
                label=row.get("label", ""),
                connections=row.get("connections", 1)
            )

        # Add edges
        edges_added = 0
        edges_skipped = 0
        for _, row in self.edges_df.iterrows():
            source = row["source"]
            target = row["target"]

            # Verify both nodes exist
            if source not in self.graph:
                edges_skipped += 1
                continue
            if target not in self.graph:
                edges_skipped += 1
                continue

            cap = row["capacity_bcm_day"]
            if pd.isna(cap) or cap <= 0:
                cap = 0.01 / 365.0  # Small default

            self.graph.add_edge(
                source, target,
                capacity=cap,
                original_capacity=cap,
                length_km=row["length_km"]
                          if not pd.isna(row["length_km"]) else 100.0,
                pipeline_name=row["pipeline_name"]
                              if not pd.isna(row["pipeline_name"])
                              else "Unknown",
                owner=row["owner"]
                      if not pd.isna(row["owner"]) else "Unknown"
            )
            edges_added += 1

        # Store original for reset
        self.original_graph = self.graph.copy()

        # Calculate and store baseline
        self.baseline_kpis = self.calculate_kpis()

        print(f"Network loaded:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {edges_added} added, {edges_skipped} skipped")
        print(f"  Connected components: "
              f"{self.baseline_kpis['n_components']}")
        print(f"  Total capacity: "
              f"{self.baseline_kpis['total_capacity']:.4f} BCM/day")

    def calculate_kpis(self):
        """
        Calculate network health Key Performance Indicators.

        Returns dict with:
        - n_components: number of connected components (1 = fully connected)
        - total_capacity: sum of all edge capacities (BCM/day)
        - edge_connectivity: minimum edges to remove to disconnect
        - n_nodes: number of nodes
        - n_edges: number of edges
        - node_betweenness: dict of node → betweenness centrality
        - edge_betweenness: dict of edge → betweenness centrality
        """
        G = self.graph

        if G.number_of_nodes() == 0:
            return {
                "n_components": 0,
                "total_capacity": 0.0,
                "edge_connectivity": 0,
                "n_nodes": 0,
                "n_edges": 0,
                "node_betweenness": {},
                "edge_betweenness": {},
            }

        # KPI 1: Connectivity
        n_components = nx.number_connected_components(G)

        # KPI 2: Total capacity
        total_cap = sum(
            d["capacity"] for _, _, d in G.edges(data=True)
        )

        # KPI 3: Edge connectivity
        if n_components == 1 and G.number_of_nodes() > 1:
            try:
                edge_conn = nx.edge_connectivity(G)
            except nx.NetworkXError:
                edge_conn = 0
        else:
            edge_conn = 0

        # KPI 4: Betweenness centrality
        if G.number_of_nodes() > 1:
            node_betweenness = nx.betweenness_centrality(G)
            edge_betweenness = nx.edge_betweenness_centrality(G)
        else:
            node_betweenness = {}
            edge_betweenness = {}

        return {
            "n_components": n_components,
            "total_capacity": total_cap,
            "edge_connectivity": edge_conn,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "node_betweenness": node_betweenness,
            "edge_betweenness": edge_betweenness,
        }

    def apply_disruption(self, disruption):
        """
        Apply a disruption scenario to the network.

        disruption dict can contain:
        - "remove_nodes": list of node IDs to completely remove
        - "reduce_edges": dict of {(source, target): multiplier}
          where multiplier 0.0 = total loss, 1.0 = no change
        - "reduce_nodes": dict of {node_id: multiplier}
          reduces capacity of ALL edges touching that node
        """
        # Remove nodes (complete failure)
        for node in disruption.get("remove_nodes", []):
            if node in self.graph:
                self.graph.remove_node(node)

        # Reduce specific edge capacities
        for edge_key, mult in disruption.get("reduce_edges", {}).items():
            u, v = edge_key
            if self.graph.has_edge(u, v):
                self.graph[u][v]["capacity"] *= max(0.0, min(1.0, mult))

        # Reduce all edges touching a node
        for node, mult in disruption.get("reduce_nodes", {}).items():
            if node in self.graph:
                mult = max(0.0, min(1.0, mult))
                for neighbor in list(self.graph.neighbors(node)):
                    self.graph[node][neighbor]["capacity"] *= mult

    def reset(self):
        """Reset network to original state (undo all disruptions)."""
        self.graph = self.original_graph.copy()

    def get_node_info(self, node_id):
        """Get all information about a specific node."""
        if node_id not in self.graph:
            return None
        data = self.graph.nodes[node_id]
        neighbors = list(self.graph.neighbors(node_id))
        edges = []
        for neighbor in neighbors:
            edge_data = self.graph[node_id][neighbor]
            edges.append({
                "neighbor": neighbor,
                "capacity": edge_data["capacity"],
                "pipeline": edge_data.get("pipeline_name", "Unknown")
            })
        return {
            "id": node_id,
            "data": data,
            "neighbors": neighbors,
            "degree": len(neighbors),
            "edges": edges
        }
```

**Testing the Digital Twin:**

```python
if __name__ == "__main__":
    twin = DigitalTwin()
    twin.load_network("data/processed/nodes.csv",
                      "data/processed/edges.csv")

    # Test baseline KPIs
    baseline = twin.calculate_kpis()
    print(f"\n=== BASELINE ===")
    print(f"Nodes: {baseline['n_nodes']}")
    print(f"Edges: {baseline['n_edges']}")
    print(f"Components: {baseline['n_components']}")
    print(f"Total capacity: {baseline['total_capacity']:.4f} BCM/day")
    print(f"Edge connectivity: {baseline['edge_connectivity']}")

    # Find the highest-degree node
    top_nodes = sorted(
        twin.graph.degree(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    print(f"\nTop 5 most connected nodes:")
    for node_id, degree in top_nodes:
        info = twin.get_node_info(node_id)
        print(f"  {node_id} (degree={degree}, "
              f"type={info['data'].get('node_type', '?')}, "
              f"label={info['data'].get('label', '?')})")

    # Test disruption: remove the top node
    top_node = top_nodes[0][0]
    print(f"\n=== DISRUPTION: Remove {top_node} ===")
    twin.apply_disruption({"remove_nodes": [top_node]})
    stressed = twin.calculate_kpis()
    print(f"Components: {baseline['n_components']} → "
          f"{stressed['n_components']}")
    print(f"Capacity: {baseline['total_capacity']:.4f} → "
          f"{stressed['total_capacity']:.4f} BCM/day")
    cap_retained = (stressed['total_capacity'] /
                    baseline['total_capacity'] * 100)
    print(f"Capacity retained: {cap_retained:.1f}%")

    # Test reset
    twin.reset()
    after_reset = twin.calculate_kpis()
    print(f"\n=== AFTER RESET ===")
    print(f"Nodes: {after_reset['n_nodes']} "
          f"(should be {baseline['n_nodes']})")
    print(f"Capacity: {after_reset['total_capacity']:.4f} "
          f"(should be {baseline['total_capacity']:.4f})")

    assert after_reset['n_nodes'] == baseline['n_nodes'], \
        "RESET FAILED: node count mismatch!"
    assert abs(after_reset['total_capacity'] -
               baseline['total_capacity']) < 0.0001, \
        "RESET FAILED: capacity mismatch!"
    print("Reset verification: PASSED")
```

```
WHAT CAN GO WRONG:

PROBLEM: "n_components: 5" at baseline (should ideally be 1)
CAUSE:   The network is disconnected — some pipelines are isolated
WHY:     Some pipelines in the dataset do not connect to others.
         This is actually normal for real data (Alaska pipeline
         does not connect to Texas pipeline).
FIX:     This is NOT a bug. The network being disconnected is a real
         finding. However, for stress testing, focus on the
         LARGEST connected component:
         
         largest_cc = max(nx.connected_components(self.graph),
                         key=len)
         subgraph = self.graph.subgraph(largest_cc).copy()
         self.graph = subgraph
         
         Add this option to load_network() with a parameter
         like largest_component_only=True

PROBLEM: edge_connectivity calculation takes very long (minutes)
CAUSE:   nx.edge_connectivity() is computationally expensive
         for large graphs (O(n² × m) in worst case)
FIX:     Skip this KPI for large graphs:
         if G.number_of_nodes() > 150:
             edge_conn = -1  # Skip, too slow
         else:
             edge_conn = nx.edge_connectivity(G)
         
         Or use approximate version:
         edge_conn = nx.minimum_edge_cut(G)
         len(edge_conn) gives the connectivity number

PROBLEM: betweenness_centrality takes very long
CAUSE:   O(n × m) computation on large graphs
FIX:     Use approximate version with k samples:
         node_betweenness = nx.betweenness_centrality(G, k=50)
         This samples 50 random nodes instead of all.
         Much faster, close enough for our purposes.

PROBLEM: reset() does not fully restore the graph
CAUSE:   graph.copy() creates a shallow copy in some NetworkX versions
FIX:     Use deep copy:
         import copy
         self.original_graph = copy.deepcopy(self.graph)
         And in reset:
         self.graph = copy.deepcopy(self.original_graph)

PROBLEM: Removing a terminal node (degree 1) causes no fragmentation
CAUSE:   This is correct behavior. Removing a leaf node does not
         split the network. Only removing a bridge node or cut vertex
         causes fragmentation.
FIX:     Not a bug. This is expected. Hub nodes with high degree
         should cause more impact.

PROBLEM: "capacity_bcm_day" column contains NaN values
CAUSE:   Some rows in edges.csv have empty capacity
FIX:     Already handled in load_network() with the pd.isna() check.
         But if it still appears, add this after loading edges_df:
         self.edges_df["capacity_bcm_day"] = \
             self.edges_df["capacity_bcm_day"].fillna(0.01/365)
```

**Done when:**
- Digital twin loads the network
- Baseline shows reasonable KPIs
- Removing a hub node increases components and decreases capacity
- Reset fully restores the original state

---

### PHASE 4: STRESS TESTER

**File:** `engine/stress_tester.py`

```python
import pandas as pd
import numpy as np
from engine.digital_twin import DigitalTwin

def find_vulnerable_nodes(twin):
    """
    Test every node by removing it and measuring impact.
    Returns ranked list of (node_id, impact_dict).
    """
    baseline = twin.calculate_kpis()
    vulnerability = {}

    all_nodes = list(twin.graph.nodes())
    total = len(all_nodes)

    for idx, node in enumerate(all_nodes):
        if idx % 20 == 0:
            print(f"  Testing node {idx+1}/{total}...")

        # Apply disruption
        twin.apply_disruption({"remove_nodes": [node]})
        stressed = twin.calculate_kpis()

        # Measure impact
        if baseline["total_capacity"] > 0:
            cap_loss_pct = (
                1 - stressed["total_capacity"] /
                baseline["total_capacity"]
            ) * 100
        else:
            cap_loss_pct = 0

        fragmentation = (stressed["n_components"] -
                         baseline["n_components"])

        # Impact score: capacity loss + fragmentation penalty
        # Fragmentation is weighted heavily because splitting
        # the network is worse than losing some capacity
        impact_score = cap_loss_pct + fragmentation * 10

        vulnerability[node] = {
            "capacity_loss_pct": round(cap_loss_pct, 2),
            "new_components": stressed["n_components"],
            "fragmentation": fragmentation,
            "impact_score": round(impact_score, 2),
            "node_type": twin.original_graph.nodes[node].get(
                "node_type", "unknown"
            ),
            "label": twin.original_graph.nodes[node].get(
                "label", ""
            ),
            "degree": twin.original_graph.degree(node),
        }

        twin.reset()

    # Sort by impact score (highest = most vulnerable)
    ranked = sorted(
        vulnerability.items(),
        key=lambda x: x[1]["impact_score"],
        reverse=True
    )

    print(f"\nTop 10 most vulnerable nodes:")
    for node_id, info in ranked[:10]:
        print(f"  {node_id}: impact={info['impact_score']:.1f}, "
              f"cap_loss={info['capacity_loss_pct']:.1f}%, "
              f"frag={info['fragmentation']}, "
              f"degree={info['degree']}, "
              f"type={info['node_type']}")

    return ranked


def find_critical_edges(twin):
    """
    Test every edge by removing it and measuring impact.
    Returns ranked list of (edge_tuple, impact_dict).
    """
    baseline = twin.calculate_kpis()
    vulnerability = {}

    all_edges = list(twin.graph.edges(data=True))
    total = len(all_edges)

    for idx, (u, v, data) in enumerate(all_edges):
        if idx % 20 == 0:
            print(f"  Testing edge {idx+1}/{total}...")

        # Remove this edge
        twin.graph.remove_edge(u, v)
        stressed = twin.calculate_kpis()

        if baseline["total_capacity"] > 0:
            cap_loss_pct = (
                1 - stressed["total_capacity"] /
                baseline["total_capacity"]
            ) * 100
        else:
            cap_loss_pct = 0

        fragmentation = (stressed["n_components"] -
                         baseline["n_components"])

        impact_score = cap_loss_pct + fragmentation * 10

        vulnerability[(u, v)] = {
            "pipeline_name": data.get("pipeline_name", "Unknown"),
            "capacity_loss_pct": round(cap_loss_pct, 2),
            "fragmentation": fragmentation,
            "impact_score": round(impact_score, 2),
        }

        twin.reset()

    ranked = sorted(
        vulnerability.items(),
        key=lambda x: x[1]["impact_score"],
        reverse=True
    )

    print(f"\nTop 10 most critical edges:")
    for edge, info in ranked[:10]:
        print(f"  {edge}: {info['pipeline_name']}, "
              f"impact={info['impact_score']:.1f}")

    return ranked


def run_batch_test(twin, scenarios):
    """
    Run multiple disruption scenarios and collect results.

    Parameters:
    -----------
    twin : DigitalTwin instance (loaded)
    scenarios : list of disruption dicts

    Returns:
    --------
    results : list of result dicts
    summary : dict with aggregate statistics
    """
    baseline = twin.calculate_kpis()
    results = []

    for i, scenario in enumerate(scenarios):
        if i % 25 == 0:
            print(f"  Running scenario {i+1}/{len(scenarios)}...")

        twin.apply_disruption(scenario)
        stressed = twin.calculate_kpis()

        if baseline["total_capacity"] > 0:
            cap_retention = (
                stressed["total_capacity"] /
                baseline["total_capacity"] * 100
            )
        else:
            cap_retention = 0

        results.append({
            "scenario_idx": i,
            "capacity_retention_pct": round(cap_retention, 2),
            "n_components": stressed["n_components"],
            "edge_connectivity": stressed["edge_connectivity"],
            "nodes_remaining": stressed["n_nodes"],
            "edges_remaining": stressed["n_edges"],
        })

        twin.reset()

    # Calculate summary statistics
    retentions = [r["capacity_retention_pct"] for r in results]
    components = [r["n_components"] for r in results]

    summary = {
        "resilience_score": round(np.mean(retentions), 1),
        "avg_capacity_retention": round(np.mean(retentions), 1),
        "min_capacity_retention": round(min(retentions), 1),
        "max_capacity_retention": round(max(retentions), 1),
        "std_capacity_retention": round(np.std(retentions), 1),
        "avg_components": round(np.mean(components), 1),
        "max_components": max(components),
        "total_scenarios": len(scenarios),
    }

    print(f"\n=== BATCH TEST SUMMARY ===")
    print(f"Resilience Score: {summary['resilience_score']}/100")
    print(f"Capacity Retention: "
          f"{summary['min_capacity_retention']}% - "
          f"{summary['max_capacity_retention']}% "
          f"(avg {summary['avg_capacity_retention']}%)")
    print(f"Worst fragmentation: {summary['max_components']} components")

    return results, summary
```

```
WHAT CAN GO WRONG:

PROBLEM: find_vulnerable_nodes takes very long (over 10 minutes)
CAUSE:   calculate_kpis() is called once per node.
         If edge_connectivity is being computed each time, and
         the graph has 200+ nodes, this is N × expensive_computation.
FIX:     In calculate_kpis(), skip edge_connectivity:
         Modify it to accept a parameter:
         def calculate_kpis(self, fast=False):
             if fast:
                 edge_conn = -1  # Skip
             else:
                 edge_conn = nx.edge_connectivity(G)
         
         Then in stress_tester, call twin.calculate_kpis(fast=True)
         for the per-node loop.

PROBLEM: All nodes have the same impact score
CAUSE:   If all edges have the same capacity (0.01 default),
         removing any node with degree D always loses D × 0.01.
         This means impact is purely proportional to degree.
FIX:     This is technically correct given uniform capacities.
         The fragmentation score adds differentiation.
         Nodes that are BRIDGES (cut vertices) will have higher
         fragmentation scores even with uniform capacities.
         You can also check: nx.bridges(G) returns all bridge edges.

PROBLEM: twin.reset() is slow (taking seconds per call)
CAUSE:   Deep copy of large graph takes time
FIX:     For the per-node test, instead of removing and resetting,
         compute metrics on a TEMPORARY copy:
         temp_graph = twin.graph.copy()
         temp_graph.remove_node(node)
         # compute KPIs on temp_graph
         # no need to reset twin.graph
```

**Testing the stress tester:**

```python
if __name__ == "__main__":
    from engine.digital_twin import DigitalTwin

    twin = DigitalTwin()
    twin.load_network("data/processed/nodes.csv",
                      "data/processed/edges.csv")

    print("\n=== FINDING VULNERABLE NODES ===")
    ranked_nodes = find_vulnerable_nodes(twin)

    print("\n=== FINDING CRITICAL EDGES ===")
    ranked_edges = find_critical_edges(twin)

    # Create some simple test scenarios
    test_scenarios = []
    top_5_nodes = [n[0] for n in ranked_nodes[:5]]
    for node in top_5_nodes:
        test_scenarios.append({"remove_nodes": [node]})

    # Also test removing pairs
    if len(top_5_nodes) >= 2:
        test_scenarios.append(
            {"remove_nodes": [top_5_nodes[0], top_5_nodes[1]]}
        )

    print(f"\n=== BATCH TEST ({len(test_scenarios)} scenarios) ===")
    results, summary = run_batch_test(twin, test_scenarios)
```

**Done when:**
- Vulnerable nodes are ranked with hub nodes near the top
- Removing a hub causes measurable fragmentation
- Batch test completes without errors
- Resilience score is computed

---

### PHASE 5: GAN TRAINING DATA

**File:** `gan/data_generator.py`

**What you are doing:** Creating the training dataset that the GAN will learn from. This is a CSV file where each row is a disruption scenario defined by regional capacity multipliers and conditions.

---

#### Phase 5A — Region Assignment

```python
import numpy as np
import pandas as pd

# US regions for gas pipeline analysis
REGIONS = {
    1: {
        "name": "Gulf Coast",
        "states": ["Texas", "Louisiana", "Mississippi",
                   "Alabama", "Florida"],
        "lat_range": (25, 33),
        "lon_range": (-100, -80),
    },
    2: {
        "name": "Northeast",
        "states": ["New York", "Pennsylvania", "New Jersey",
                   "Connecticut", "Massachusetts", "Maine",
                   "New Hampshire", "Vermont", "Rhode Island",
                   "Maryland", "Delaware", "Virginia"],
        "lat_range": (37, 47),
        "lon_range": (-80, -67),
    },
    3: {
        "name": "Midwest",
        "states": ["Ohio", "Indiana", "Illinois", "Michigan",
                   "Wisconsin", "Minnesota", "Iowa", "Missouri",
                   "Kansas", "Nebraska"],
        "lat_range": (36, 49),
        "lon_range": (-104, -80),
    },
    4: {
        "name": "West",
        "states": ["California", "Nevada", "Arizona", "Oregon",
                   "Washington", "Utah", "Idaho"],
        "lat_range": (32, 49),
        "lon_range": (-125, -104),
    },
    5: {
        "name": "Permian Basin",
        "states": [],  # Identified by coordinates
        "lat_range": (30, 34),
        "lon_range": (-105, -100),
    },
    6: {
        "name": "Appalachian",
        "states": ["West Virginia", "Kentucky", "Tennessee"],
        "lat_range": (35, 41),
        "lon_range": (-86, -78),
    },
    7: {
        "name": "Rocky Mountain",
        "states": ["Wyoming", "Montana", "North Dakota",
                   "South Dakota", "Colorado"],
        "lat_range": (39, 49),
        "lon_range": (-112, -96),
    },
}


def assign_node_region(lat, lon, state=""):
    """
    Assign a node to a region based on state name or coordinates.
    Returns region number (1-7) or 0 if unmatched.
    """
    # First try matching by state name
    for region_id, region in REGIONS.items():
        if state in region["states"]:
            return region_id

    # If state did not match, try coordinates
    for region_id, region in REGIONS.items():
        lat_min, lat_max = region["lat_range"]
        lon_min, lon_max = region["lon_range"]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return region_id

    # Default: assign to closest region by distance
    return 1  # Gulf Coast as default (largest US gas region)
```

```
WHAT CAN GO WRONG:

PROBLEM: State names in your data do not match the REGIONS dict
CAUSE:   Data might use abbreviations ("TX" vs "Texas") or
         different formatting
FIX:     Add abbreviation mapping:
         STATE_ABBREV = {"TX": "Texas", "LA": "Louisiana", ...}
         state = STATE_ABBREV.get(state, state)
         
         OR: Rely only on coordinate-based assignment
         (remove the state matching and only use lat/lon ranges)

PROBLEM: Some nodes do not match any region
CAUSE:   Alaska, Hawaii, or offshore pipelines
FIX:     The default assignment handles this. 
         Alaska nodes will default to region 1 which is incorrect
         but harmless since there are very few of them.
         Alternatively, add an "Other" region 8.

PROBLEM: Overlapping region boundaries
CAUSE:   Some states appear in multiple regions
         (Ohio is in both Midwest and Appalachian ranges)
FIX:     State matching takes priority over coordinate matching.
         If a state is listed in a region, it goes there regardless
         of coordinates. This is already handled by checking
         state first.
```

---

#### Phase 5B — Create Real Event Vectors

```python
def create_real_events():
    """
    Create disruption vectors based on real documented events.
    Each event specifies regional capacity multipliers.
    
    Multiplier meaning:
      1.0 = full capacity (no impact)
      0.5 = 50% capacity remaining
      0.0 = complete shutdown
    """
    events = [
        {
            "name": "Hurricane_Katrina_2005",
            "type": "hurricane",
            "severity": 0.9,
            "target_region": 1,
            "r1": 0.15, "r2": 0.90, "r3": 0.85, "r4": 1.00,
            "r5": 0.70, "r6": 0.95, "r7": 1.00,
            "cross": 0.60, "demand": 0.80, "price": 0.85,
            "infra_age": 0.7,
        },
        {
            "name": "Hurricane_Harvey_2017",
            "type": "hurricane",
            "severity": 0.8,
            "target_region": 1,
            "r1": 0.25, "r2": 0.95, "r3": 0.90, "r4": 1.00,
            "r5": 0.60, "r6": 0.95, "r7": 1.00,
            "cross": 0.65, "demand": 0.85, "price": 0.70,
            "infra_age": 0.6,
        },
        {
            "name": "Winter_Storm_Uri_2021",
            "type": "hurricane",  # weather event
            "severity": 0.85,
            "target_region": 1,
            "r1": 0.20, "r2": 0.70, "r3": 0.65, "r4": 0.90,
            "r5": 0.15, "r6": 0.75, "r7": 0.80,
            "cross": 0.40, "demand": 0.30, "price": 0.95,
            "infra_age": 0.8,
        },
        {
            "name": "Colonial_Pipeline_Hack_2021",
            "type": "sabotage",
            "severity": 0.7,
            "target_region": 1,
            "r1": 0.40, "r2": 0.70, "r3": 0.90, "r4": 1.00,
            "r5": 0.95, "r6": 0.85, "r7": 1.00,
            "cross": 0.50, "demand": 0.95, "price": 0.50,
            "infra_age": 0.5,
        },
        {
            "name": "Nord_Stream_Sabotage_2022",
            "type": "sabotage",
            "severity": 1.0,
            "target_region": 2,
            "r1": 0.95, "r2": 0.30, "r3": 0.80, "r4": 1.00,
            "r5": 1.00, "r6": 0.85, "r7": 1.00,
            "cross": 0.70, "demand": 0.60, "price": 0.90,
            "infra_age": 0.3,
        },
        {
            "name": "Russia_Gas_Cutoff_2022",
            "type": "sanctions",
            "severity": 0.9,
            "target_region": 2,
            "r1": 0.85, "r2": 0.40, "r3": 0.70, "r4": 0.95,
            "r5": 0.90, "r6": 0.65, "r7": 0.95,
            "cross": 0.60, "demand": 0.50, "price": 0.95,
            "infra_age": 0.4,
        },
        {
            "name": "COVID_Demand_Drop_2020",
            "type": "demand_surge",  # inverse surge
            "severity": 0.5,
            "target_region": 3,
            "r1": 0.85, "r2": 0.80, "r3": 0.75, "r4": 0.80,
            "r5": 0.70, "r6": 0.80, "r7": 0.85,
            "cross": 0.80, "demand": 0.40, "price": 0.10,
            "infra_age": 0.5,
        },
        {
            "name": "Summer_Heat_Wave_Demand_2023",
            "type": "demand_surge",
            "severity": 0.6,
            "target_region": 4,
            "r1": 0.90, "r2": 0.85, "r3": 0.90, "r4": 0.80,
            "r5": 0.95, "r6": 0.90, "r7": 0.95,
            "cross": 0.85, "demand": 0.30, "price": 0.60,
            "infra_age": 0.5,
        },
        {
            "name": "Permian_Basin_Bottleneck_2019",
            "type": "demand_surge",
            "severity": 0.65,
            "target_region": 5,
            "r1": 0.75, "r2": 0.95, "r3": 0.95, "r4": 0.95,
            "r5": 0.30, "r6": 0.95, "r7": 0.95,
            "cross": 0.50, "demand": 0.45, "price": 0.55,
            "infra_age": 0.6,
        },
        {
            "name": "Appalachian_Pipe_Constraint",
            "type": "sabotage",
            "severity": 0.55,
            "target_region": 6,
            "r1": 0.95, "r2": 0.70, "r3": 0.85, "r4": 1.00,
            "r5": 0.95, "r6": 0.35, "r7": 0.95,
            "cross": 0.65, "demand": 0.80, "price": 0.55,
            "infra_age": 0.7,
        },
        {
            "name": "Northeast_Polar_Vortex_2014",
            "type": "demand_surge",
            "severity": 0.75,
            "target_region": 2,
            "r1": 0.90, "r2": 0.50, "r3": 0.65, "r4": 0.95,
            "r5": 0.90, "r6": 0.60, "r7": 0.80,
            "cross": 0.55, "demand": 0.25, "price": 0.80,
            "infra_age": 0.6,
        },
        {
            "name": "Gulf_Hurricane_Cat3_Generic",
            "type": "hurricane",
            "severity": 0.7,
            "target_region": 1,
            "r1": 0.35, "r2": 0.95, "r3": 0.90, "r4": 1.00,
            "r5": 0.75, "r6": 0.95, "r7": 1.00,
            "cross": 0.70, "demand": 0.85, "price": 0.55,
            "infra_age": 0.5,
        },
        {
            "name": "Cyber_Attack_Midwest_Grid",
            "type": "sabotage",
            "severity": 0.6,
            "target_region": 3,
            "r1": 0.90, "r2": 0.85, "r3": 0.25, "r4": 0.95,
            "r5": 0.90, "r6": 0.70, "r7": 0.90,
            "cross": 0.55, "demand": 0.90, "price": 0.55,
            "infra_age": 0.4,
        },
        {
            "name": "Earthquake_West_Coast",
            "type": "hurricane",
            "severity": 0.7,
            "target_region": 4,
            "r1": 0.95, "r2": 0.95, "r3": 0.95, "r4": 0.20,
            "r5": 0.90, "r6": 0.95, "r7": 0.85,
            "cross": 0.60, "demand": 0.75, "price": 0.65,
            "infra_age": 0.6,
        },
        {
            "name": "Multi_Factor_Winter_Storm_Plus_Demand",
            "type": "multi_factor",
            "severity": 0.95,
            "target_region": 1,
            "r1": 0.10, "r2": 0.55, "r3": 0.50, "r4": 0.80,
            "r5": 0.15, "r6": 0.60, "r7": 0.75,
            "cross": 0.30, "demand": 0.20, "price": 0.95,
            "infra_age": 0.9,
        },
        {
            "name": "Multi_Factor_Sanctions_Plus_Hurricane",
            "type": "multi_factor",
            "severity": 0.85,
            "target_region": 1,
            "r1": 0.20, "r2": 0.60, "r3": 0.70, "r4": 0.90,
            "r5": 0.50, "r6": 0.70, "r7": 0.90,
            "cross": 0.40, "demand": 0.45, "price": 0.90,
            "infra_age": 0.7,
        },
        {
            "name": "LNG_Export_Surge",
            "type": "demand_surge",
            "severity": 0.5,
            "target_region": 1,
            "r1": 0.70, "r2": 0.90, "r3": 0.90, "r4": 0.95,
            "r5": 0.80, "r6": 0.90, "r7": 0.95,
            "cross": 0.75, "demand": 0.35, "price": 0.50,
            "infra_age": 0.4,
        },
        {
            "name": "Rockies_Wildfire",
            "type": "hurricane",
            "severity": 0.5,
            "target_region": 7,
            "r1": 0.95, "r2": 0.95, "r3": 0.90, "r4": 0.85,
            "r5": 0.95, "r6": 0.95, "r7": 0.30,
            "cross": 0.70, "demand": 0.85, "price": 0.40,
            "infra_age": 0.5,
        },
    ]

    return events
```

---

#### Phase 5C — Augmentation and Final Dataset

```python
def create_training_dataset(events, variations_per_event=250,
                            noise_std=0.08):
    """
    Augment real events with noise variations.
    Also add price-derived scenarios from yfinance.
    """
    # Type encoding
    type_map = {
        "hurricane": [1, 0, 0, 0, 0],
        "sabotage": [0, 1, 0, 0, 0],
        "sanctions": [0, 0, 1, 0, 0],
        "demand_surge": [0, 0, 0, 1, 0],
        "multi_factor": [0, 0, 0, 0, 1],
    }

    output_keys = ["r1", "r2", "r3", "r4", "r5", "r6", "r7",
                   "cross", "demand", "price", "infra_age"]

    all_rows = []

    # Part 1: Augmented real events
    for event in events:
        type_vec = type_map.get(event["type"], [0, 0, 0, 0, 1])
        severity = event["severity"]
        target_region = event["target_region"] / 7.0  # Normalize

        base_outputs = [event[k] for k in output_keys]

        for v in range(variations_per_event):
            # Add Gaussian noise
            noisy_outputs = []
            for val in base_outputs:
                noisy_val = val + np.random.normal(0, noise_std)
                noisy_val = np.clip(noisy_val, 0.0, 1.0)
                noisy_outputs.append(round(noisy_val, 4))

            # Slight severity variation
            noisy_severity = np.clip(
                severity + np.random.normal(0, 0.05), 0.0, 1.0
            )

            row = (type_vec +
                   [round(noisy_severity, 4),
                    round(target_region, 4)] +
                   noisy_outputs)
            all_rows.append(row)

    # Part 2: Price-derived scenarios
    try:
        import yfinance as yf
        print("Downloading Henry Hub price data...")
        ng = yf.download("NG=F", start="2015-01-01",
                         progress=False)
        if len(ng) > 0:
            prices = ng["Close"].dropna()
            rolling_mean = prices.rolling(60).mean()
            deviation = (prices - rolling_mean) / rolling_mean

            # Find months with >15% deviation
            extreme = deviation[abs(deviation) > 0.15].dropna()
            print(f"  Found {len(extreme)} extreme price periods")

            for idx, dev_val in extreme.items():
                dev = float(dev_val)
                severity = min(abs(dev), 1.0)
                is_spike = dev > 0

                if is_spike:
                    # Price spike = supply constraint somewhere
                    type_vec = [0, 0, 0, 1, 0]  # demand_surge
                    base = [
                        np.random.uniform(0.7, 0.95),  # r1-r7
                        np.random.uniform(0.7, 0.95),
                        np.random.uniform(0.7, 0.95),
                        np.random.uniform(0.7, 0.95),
                        np.random.uniform(0.7, 0.95),
                        np.random.uniform(0.7, 0.95),
                        np.random.uniform(0.7, 0.95),
                        np.random.uniform(0.75, 0.95),  # cross
                        np.random.uniform(0.2, 0.5),     # demand
                        min(abs(dev), 1.0),              # price
                        0.5,                             # infra_age
                    ]
                else:
                    # Price crash = demand drop
                    type_vec = [0, 0, 0, 1, 0]
                    base = [
                        np.random.uniform(0.80, 1.0),
                        np.random.uniform(0.80, 1.0),
                        np.random.uniform(0.80, 1.0),
                        np.random.uniform(0.80, 1.0),
                        np.random.uniform(0.80, 1.0),
                        np.random.uniform(0.80, 1.0),
                        np.random.uniform(0.80, 1.0),
                        np.random.uniform(0.85, 1.0),
                        np.random.uniform(0.5, 0.8),
                        max(0.0, 1.0 - abs(dev)),
                        0.5,
                    ]

                row = (type_vec +
                       [round(severity, 4), 0.1429] +  # region=1/7
                       [round(v, 4) for v in base])
                all_rows.append(row)
        else:
            print("  WARNING: No price data downloaded. Skipping.")
    except Exception as e:
        print(f"  WARNING: yfinance failed: {e}")
        print(f"  Continuing without price data.")

    # Build DataFrame
    columns = (
        ["type_hurricane", "type_sabotage", "type_sanctions",
         "type_demand", "type_multi",
         "severity", "target_region"] +
        ["r1_cap", "r2_cap", "r3_cap", "r4_cap", "r5_cap",
         "r6_cap", "r7_cap",
         "cross_cap", "demand_mult", "price_mult", "infra_age"]
    )

    df = pd.DataFrame(all_rows, columns=columns)

    # Save
    df.to_csv("data/training/disruption_scenarios.csv", index=False)
    print(f"\nTraining data saved: {len(df)} scenarios")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Shape: {df.shape}")

    return df


if __name__ == "__main__":
    events = create_real_events()
    print(f"Created {len(events)} base events")
    df = create_training_dataset(events)
```

```
WHAT CAN GO WRONG:

PROBLEM: yfinance download fails with "No data found"
CAUSE:   Network issues, yfinance API changes, or ticker symbol wrong
FIX:     The code already handles this with try/except.
         The price data is a bonus, not required.
         If it fails, you still have 4,500+ scenarios from events.
         You can also try: ticker "NG=F" might need to be "NG=F"
         or "HH=F". Check yfinance documentation.

PROBLEM: yfinance returns empty DataFrame
CAUSE:   Yahoo Finance API may have changed
FIX:     Same as above — skip it. The event-based data is enough.

PROBLEM: All values in the CSV are the same
CAUSE:   Noise std is too small or there is a bug in the loop
FIX:     Check: df.describe() should show variation in all columns.
         Increase noise_std to 0.10 or 0.12 if values are too similar.

PROBLEM: Training data has NaN values
CAUSE:   Some computation produced NaN
FIX:     df = df.fillna(0.5)  # Fill with neutral value
         df = df.dropna()     # Or drop rows with NaN
```

**Done when:** `data/training/disruption_scenarios.csv` exists with 4,500+ rows and 18 columns, no NaN values, all values between 0 and 1.

---

### PHASE 6: GAN MODEL AND TRAINING

---

#### Phase 6A — Model Architecture

**File:** `gan/model.py`

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=32, c_dim=7, output_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, output_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=11, c_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + c_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, c):
        inp = torch.cat([x, c], dim=1)
        return self.net(inp)
```

```
WHAT CAN GO WRONG:

PROBLEM: BatchNorm1d crashes with "Expected more than 1 value per channel"
CAUSE:   Batch size is 1. BatchNorm needs batch size >= 2.
FIX:     Ensure batch_size >= 2 in training.
         During generation (batch_size=1), use model.eval() mode:
         generator.eval()
         with torch.no_grad():
             output = generator(z, c)
         BatchNorm behaves differently in eval mode.

PROBLEM: Dimensions mismatch error
CAUSE:   z_dim + c_dim does not match the Linear input
FIX:     z_dim=32, c_dim=7, so first Linear should be 39.
         output_dim=11, input_dim for discriminator should be 11+7=18.
         Double check these match your training data columns.
         c_dim = 7 (5 type one-hot + 1 severity + 1 target_region)
         output_dim = 11 (7 regional + cross + demand + price + infra_age)
```

---

#### Phase 6B — Training Loop

**File:** `gan/train.py`

```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from gan.model import Generator, Discriminator

def train_gan(data_path="data/training/disruption_scenarios.csv",
              epochs=500, batch_size=64, lr=0.0002,
              z_dim=32, c_dim=7, output_dim=11):
    """Train the conditional GAN on disruption scenarios."""

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} training scenarios")

    # Split into condition (first 7 cols) and output (last 11 cols)
    condition_cols = df.columns[:c_dim].tolist()
    output_cols = df.columns[c_dim:c_dim + output_dim].tolist()

    print(f"Condition columns ({c_dim}): {condition_cols}")
    print(f"Output columns ({output_dim}): {output_cols}")

    # Verify column counts
    if len(condition_cols) != c_dim:
        raise ValueError(
            f"Expected {c_dim} condition columns, "
            f"got {len(condition_cols)}"
        )
    if len(output_cols) != output_dim:
        raise ValueError(
            f"Expected {output_dim} output columns, "
            f"got {len(output_cols)}"
        )

    conditions = torch.FloatTensor(df[condition_cols].values)
    outputs = torch.FloatTensor(df[output_cols].values)

    # Check for NaN
    if torch.isnan(conditions).any():
        print("WARNING: NaN in conditions! Replacing with 0.")
        conditions = torch.nan_to_num(conditions, nan=0.0)
    if torch.isnan(outputs).any():
        print("WARNING: NaN in outputs! Replacing with 0.5.")
        outputs = torch.nan_to_num(outputs, nan=0.5)

    # Initialize models
    G = Generator(z_dim, c_dim, output_dim)
    D = Discriminator(output_dim, c_dim)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = torch.utils.data.TensorDataset(outputs, conditions)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True  # Important: avoids batch_size=1 for BatchNorm
    )

    print(f"\nTraining for {epochs} epochs...")
    print(f"Batches per epoch: {len(loader)}")

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        n_batches = 0

        for real_data, cond in loader:
            batch = real_data.size(0)

            # Labels with smoothing
            real_labels = torch.full((batch, 1), 0.9)
            fake_labels = torch.full((batch, 1), 0.1)

            # ── Train Discriminator ──
            z = torch.randn(batch, z_dim)
            fake_data = G(z, cond)

            d_real = D(real_data, cond)
            d_fake = D(fake_data.detach(), cond)

            loss_d = (criterion(d_real, real_labels) +
                      criterion(d_fake, fake_labels)) / 2

            opt_D.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

            # ── Train Generator ──
            z = torch.randn(batch, z_dim)
            fake_data = G(z, cond)
            d_fake = D(fake_data, cond)

            loss_g = criterion(d_fake, real_labels)  # Fool D

            opt_G.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            n_batches += 1

        avg_g = epoch_g_loss / max(n_batches, 1)
        avg_d = epoch_d_loss / max(n_batches, 1)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}/{epochs}: "
                  f"G_loss={avg_g:.4f}, D_loss={avg_d:.4f}")

    # Save models
    torch.save(G.state_dict(), "gan/generator.pth")
    torch.save(D.state_dict(), "gan/discriminator.pth")
    print(f"\nModels saved to gan/generator.pth and gan/discriminator.pth")

    # Save loss history
    loss_df = pd.DataFrame({
        "epoch": range(epochs),
        "g_loss": g_losses,
        "d_loss": d_losses
    })
    loss_df.to_csv("gan/training_losses.csv", index=False)

    return G, D


if __name__ == "__main__":
    train_gan()
```

```
WHAT CAN GO WRONG:

PROBLEM: G_loss stays at exactly 0.6931 (ln(2)) and never changes
CAUSE:   Generator is producing outputs that D rates as exactly 50%
         real. This means neither model is learning.
FIX 1:   Lower learning rate to 0.0001
FIX 2:   Increase network capacity (add more hidden units: 256, 512)
FIX 3:   Check that your training data has actual variation:
         print(df.describe()) — values should NOT all be the same

PROBLEM: D_loss goes to 0 very quickly and G_loss explodes
CAUSE:   Discriminator is too strong, generator cannot fool it
FIX 1:   Train G twice for every D training step
         (add another G training block inside the loop)
FIX 2:   Add more dropout to D (increase from 0.3 to 0.5)
FIX 3:   Use label smoothing more aggressively (real=0.85)

PROBLEM: Both losses oscillate wildly
CAUSE:   Learning rate too high
FIX:     Reduce lr to 0.0001 or 0.00005

PROBLEM: "RuntimeError: expected scalar type Float but found Double"
CAUSE:   Data is float64 (double) but PyTorch expects float32
FIX:     Already handled by using torch.FloatTensor() above.
         If it still happens:
         conditions = conditions.float()
         outputs = outputs.float()

PROBLEM: Training takes very long (more than 30 minutes)
CAUSE:   Dataset is large or too many epochs
FIX:     For 5000 scenarios, 500 epochs should take 2-5 minutes.
         If longer, reduce epochs to 300.
         If still too slow, check if GPU is available and move
         models to GPU:
         device = torch.device("cuda" if torch.cuda.is_available()
                               else "cpu")
         G = G.to(device)
         D = D.to(device)
         (Also move data tensors to device in the training loop)

PROBLEM: "ValueError: Expected more than 1 value per channel"
CAUSE:   Last batch has only 1 sample and BatchNorm fails
FIX:     Already handled with drop_last=True in DataLoader.
         If it still happens, use batch_size that evenly divides
         the dataset size, or add:
         if batch < 2: continue  (skip tiny batches)
```

---

#### Phase 6C — Scenario Generation

**File:** `gan/generate.py`

```python
import torch
import numpy as np
import pandas as pd
from gan.model import Generator

def load_generator(model_path="gan/generator.pth",
                   z_dim=32, c_dim=7, output_dim=11):
    """Load a trained generator."""
    G = Generator(z_dim, c_dim, output_dim)
    G.load_state_dict(torch.load(model_path,
                                  map_location="cpu"))
    G.eval()
    return G


def generate_scenarios(G, disruption_type, severity,
                       target_region, n_scenarios=100,
                       z_dim=32):
    """
    Generate n disruption scenarios from the trained GAN.

    Parameters:
    -----------
    disruption_type : str, one of "hurricane", "sabotage",
                      "sanctions", "demand_surge", "multi_factor"
    severity : float, 0.0 to 1.0
    target_region : int, 1-7
    n_scenarios : int

    Returns:
    --------
    list of dicts, each containing regional capacity multipliers
    """
    type_map = {
        "hurricane": [1, 0, 0, 0, 0],
        "sabotage": [0, 1, 0, 0, 0],
        "sanctions": [0, 0, 1, 0, 0],
        "demand_surge": [0, 0, 0, 1, 0],
        "multi_factor": [0, 0, 0, 0, 1],
    }

    type_vec = type_map.get(disruption_type, [0, 0, 0, 0, 1])
    region_norm = target_region / 7.0

    # Build condition vector
    c = torch.FloatTensor(
        [type_vec + [severity, region_norm]]
    ).repeat(n_scenarios, 1)

    # Generate
    z = torch.randn(n_scenarios, z_dim)
    with torch.no_grad():
        raw_output = G(z, c).numpy()

    # Convert to named scenarios
    scenarios = []
    for i in range(n_scenarios):
        out = raw_output[i]
        scenario = {
            "r1_cap": float(np.clip(out[0], 0, 1)),
            "r2_cap": float(np.clip(out[1], 0, 1)),
            "r3_cap": float(np.clip(out[2], 0, 1)),
            "r4_cap": float(np.clip(out[3], 0, 1)),
            "r5_cap": float(np.clip(out[4], 0, 1)),
            "r6_cap": float(np.clip(out[5], 0, 1)),
            "r7_cap": float(np.clip(out[6], 0, 1)),
            "cross_cap": float(np.clip(out[7], 0, 1)),
            "demand_mult": float(0.5 + np.clip(out[8], 0, 1)),
            "price_mult": float(1.0 + np.clip(out[9], 0, 1) * 4.0),
            "infra_age": float(np.clip(out[10], 0, 1)),
            "type": disruption_type,
            "severity": severity,
            "target_region": target_region,
        }
        scenarios.append(scenario)

    return scenarios


def scenario_to_disruption(scenario, nodes_df, edges_df,
                           region_assignments):
    """
    Convert a GAN-generated scenario (regional multipliers)
    into a disruption dict that the DigitalTwin can apply.

    Parameters:
    -----------
    scenario : dict with r1_cap through r7_cap and cross_cap
    nodes_df : DataFrame with node id, lat, lon, etc.
    edges_df : DataFrame with source, target, etc.
    region_assignments : dict of {node_id: region_number}

    Returns:
    --------
    disruption dict with "reduce_nodes" and "reduce_edges"
    """
    region_caps = {
        1: scenario["r1_cap"],
        2: scenario["r2_cap"],
        3: scenario["r3_cap"],
        4: scenario["r4_cap"],
        5: scenario["r5_cap"],
        6: scenario["r6_cap"],
        7: scenario["r7_cap"],
    }

    # Reduce nodes based on their region
    reduce_nodes = {}
    for node_id, region in region_assignments.items():
        mult = region_caps.get(region, 1.0)
        if mult < 0.99:  # Only include if actually reduced
            reduce_nodes[node_id] = mult

    # Reduce cross-region edges
    reduce_edges = {}
    for _, row in edges_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        src_region = region_assignments.get(src, 0)
        tgt_region = region_assignments.get(tgt, 0)

        if src_region != tgt_region and src_region > 0 and tgt_region > 0:
            reduce_edges[(src, tgt)] = scenario["cross_cap"]

    disruption = {
        "reduce_nodes": reduce_nodes,
        "reduce_edges": reduce_edges,
    }

    return disruption


def build_region_assignments(nodes_df):
    """Assign each node to a US region based on coordinates/state."""
    assignments = {}
    for _, row in nodes_df.iterrows():
        region = assign_node_region(
            row["lat"], row["lon"],
            row.get("label", "")
        )
        assignments[row["id"]] = region
    return assignments
```

```
WHAT CAN GO WRONG:

PROBLEM: Generator outputs are all nearly identical
CAUSE:   GAN mode collapse — generator found one output that fools
         discriminator and keeps producing it
FIX 1:   Check training losses. If D_loss ≈ 0 and G_loss is high,
         discriminator won. Retrain with more dropout in D.
FIX 2:   Add noise to real data during D training:
         real_data += torch.randn_like(real_data) * 0.05
FIX 3:   Use Wasserstein loss instead of BCE (more advanced)
FIX 4:   Accept it and add noise to generated scenarios:
         for key in scenario:
             if "cap" in key:
                 scenario[key] += np.random.normal(0, 0.1)
                 scenario[key] = np.clip(scenario[key], 0, 1)

PROBLEM: Generated values are all 0 or all 1 (saturated sigmoid)
CAUSE:   Sigmoid output is saturating
FIX:     Use Tanh instead of Sigmoid and rescale:
         Output → Tanh → range [-1, 1] → rescale to [0, 1]
         out = (self.net(x) + 1) / 2

PROBLEM: scenario_to_disruption produces empty disruption dict
CAUSE:   region_assignments is empty or does not match node IDs
FIX:     Print region_assignments to verify it has entries.
         Check that node IDs in nodes_df match those in the graph.

PROBLEM: torch.load() fails with "No such file"
CAUSE:   Model was not saved, or path is wrong
FIX:     Check that gan/generator.pth exists.
         Run train.py first. Check the save path.
```

---

### PHASE 7: INTEGRATION TEST

Before building the dashboard, test the complete pipeline end to end:

```python
# test_integration.py
from engine.data_processor import assign_node_region
from engine.digital_twin import DigitalTwin
from engine.stress_tester import run_batch_test
from gan.generate import (load_generator, generate_scenarios,
                          scenario_to_disruption)
import pandas as pd

# Step 1: Load the twin
print("=== Step 1: Loading Digital Twin ===")
twin = DigitalTwin()
twin.load_network("data/processed/nodes.csv",
                  "data/processed/edges.csv")

# Step 2: Build region assignments
print("\n=== Step 2: Assigning Regions ===")
nodes_df = pd.read_csv("data/processed/nodes.csv")
edges_df = pd.read_csv("data/processed/edges.csv")

region_assignments = {}
for _, row in nodes_df.iterrows():
    region = assign_node_region(
        row["lat"], row["lon"], row.get("label", "")
    )
    region_assignments[row["id"]] = region

region_counts = {}
for r in region_assignments.values():
    region_counts[r] = region_counts.get(r, 0) + 1
print(f"Region distribution: {region_counts}")

# Step 3: Load GAN and generate scenarios
print("\n=== Step 3: Generating Scenarios ===")
G = load_generator("gan/generator.pth")
scenarios_raw = generate_scenarios(
    G, "hurricane", severity=0.8,
    target_region=1, n_scenarios=50
)
print(f"Generated {len(scenarios_raw)} raw scenarios")
print(f"Sample scenario: {scenarios_raw[0]}")

# Step 4: Convert to disruptions
print("\n=== Step 4: Converting to Disruptions ===")
disruptions = []
for s in scenarios_raw:
    d = scenario_to_disruption(s, nodes_df, edges_df,
                                region_assignments)
    disruptions.append(d)

print(f"First disruption has {len(disruptions[0]['reduce_nodes'])} "
      f"node reductions and {len(disruptions[0]['reduce_edges'])} "
      f"edge reductions")

# Step 5: Run stress tests
print("\n=== Step 5: Running Stress Tests ===")
results, summary = run_batch_test(twin, disruptions)

print(f"\n=== FINAL RESULTS ===")
print(f"Resilience Score: {summary['resilience_score']}/100")
print(f"Average Capacity Retained: "
      f"{summary['avg_capacity_retention']}%")
print(f"Worst Case: {summary['min_capacity_retention']}% capacity, "
      f"{summary['max_components']} components")

# Step 6: Verify results make sense
print("\n=== SANITY CHECKS ===")
assert summary['resilience_score'] > 0, \
    "FAIL: Resilience should be > 0"
assert summary['resilience_score'] < 100, \
    "FAIL: Resilience should be < 100 (some disruption expected)"
assert summary['avg_capacity_retention'] > 10, \
    "FAIL: Average capacity should be > 10% (not total wipeout)"
assert summary['avg_capacity_retention'] < 100, \
    "FAIL: Should be < 100% (disruption should have effect)"
print("All sanity checks PASSED")
```

```
WHAT CAN GO WRONG AT INTEGRATION:

PROBLEM: Everything passes but resilience is always 95-100%
CAUSE:   Disruptions are not actually affecting the network.
         reduce_nodes multipliers might all be very close to 1.0.
FIX:     Print the actual multipliers:
         for node, mult in disruptions[0]["reduce_nodes"].items():
             print(f"  {node}: {mult}")
         If multipliers are > 0.95, the GAN is generating mild
         scenarios. This might mean training data was too mild.
         Try generating with higher severity (0.95) or check
         that training data has truly low values (0.1-0.3).

PROBLEM: Resilience is always 0-5%
CAUSE:   All nodes are being set to near-zero capacity.
FIX:     Same as above — check multipliers.
         Also check that "reduce_nodes" is applying multiplication
         not setting absolute values.

PROBLEM: Import errors when running integration test
CAUSE:   Module path issues
FIX:     Run from the project root directory.
         Use: python -m test_integration (if saved as module)
         Or add to top of test file:
         import sys
         sys.path.insert(0, ".")
```

**Done when:** Integration test passes all sanity checks with resilience score between 20 and 90.

---

### PHASE 8: STREAMLIT DASHBOARD

**File:** `app.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(__file__))

from engine.digital_twin import DigitalTwin
from engine.stress_tester import (find_vulnerable_nodes,
                                   run_batch_test)
from engine.data_processor import assign_node_region
from gan.generate import (load_generator, generate_scenarios,
                          scenario_to_disruption)

st.set_page_config(
    page_title="Gas Pipeline Stress Tester",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Global Gas Pipeline Network Stress Tester")
st.markdown("Stress testing US gas pipeline infrastructure "
            "using real data + GAN-generated disruption scenarios")


@st.cache_resource
def load_twin():
    """Load the digital twin (cached so it only loads once)."""
    twin = DigitalTwin()
    twin.load_network("data/processed/nodes.csv",
                      "data/processed/edges.csv")
    return twin


@st.cache_resource
def load_gan():
    """Load the trained GAN generator."""
    try:
        G = load_generator("gan/generator.pth")
        return G
    except FileNotFoundError:
        st.error("GAN model not found! Run gan/train.py first.")
        return None


@st.cache_data
def load_data():
    """Load processed data files."""
    nodes = pd.read_csv("data/processed/nodes.csv")
    edges = pd.read_csv("data/processed/edges.csv")
    return nodes, edges


@st.cache_data
def get_region_assignments(_nodes_df):
    """Assign regions to all nodes."""
    assignments = {}
    for _, row in _nodes_df.iterrows():
        region = assign_node_region(
            row["lat"], row["lon"], row.get("label", "")
        )
        assignments[row["id"]] = region
    return assignments


# Load everything
twin = load_twin()
nodes_df, edges_df = load_data()
region_assignments = get_region_assignments(nodes_df)
G = load_gan()

# ── SIDEBAR CONTROLS ──
st.sidebar.header("Stress Test Configuration")

disruption_type = st.sidebar.selectbox(
    "Disruption Type",
    ["hurricane", "sabotage", "sanctions",
     "demand_surge", "multi_factor"],
    index=0
)

severity = st.sidebar.slider(
    "Severity", 0.0, 1.0, 0.7, 0.05
)

target_region = st.sidebar.selectbox(
    "Target Region",
    {1: "Gulf Coast", 2: "Northeast", 3: "Midwest",
     4: "West", 5: "Permian Basin", 6: "Appalachian",
     7: "Rocky Mountain"},
    format_func=lambda x: {
        1: "1 - Gulf Coast", 2: "2 - Northeast",
        3: "3 - Midwest", 4: "4 - West",
        5: "5 - Permian Basin", 6: "6 - Appalachian",
        7: "7 - Rocky Mountain"
    }[x]
)

n_scenarios = st.sidebar.slider(
    "Number of Scenarios", 10, 200, 50, 10
)

run_test = st.sidebar.button("🚀 Run Stress Test",
                              type="primary")

# ── TABS ──
tab1, tab2, tab3 = st.tabs(
    ["🗺️ Pipeline Map", "📊 Stress Results", "🏢 Owner Analysis"]
)

# ── TAB 1: MAP ──
with tab1:
    st.subheader("US Gas Pipeline Network")

    fig_map = go.Figure()

    # Draw pipeline routes
    for _, edge in edges_df.iterrows():
        geom_str = edge.get("geometry", "")
        if geom_str and geom_str != "":
            try:
                geom = json.loads(str(geom_str))
                for segment in geom:
                    if isinstance(segment, list) and len(segment) > 0:
                        # Check if segment is list of [lon, lat] pairs
                        if isinstance(segment[0], list):
                            lons = [p[0] for p in segment]
                            lats = [p[1] for p in segment]
                        else:
                            # Might be a single point
                            continue

                        fig_map.add_trace(go.Scattergeo(
                            lon=lons, lat=lats,
                            mode="lines",
                            line=dict(width=1.5, color="steelblue"),
                            showlegend=False,
                            hovertext=edge.get("pipeline_name", ""),
                            hoverinfo="text"
                        ))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass  # Skip pipelines with bad geometry

    # Draw hub nodes
    hub_nodes = nodes_df[nodes_df["type"] == "hub"]
    if len(hub_nodes) > 0:
        fig_map.add_trace(go.Scattergeo(
            lon=hub_nodes["lon"],
            lat=hub_nodes["lat"],
            mode="markers+text",
            marker=dict(size=10, color="red",
                        symbol="circle"),
            text=hub_nodes["label"],
            textposition="top center",
            textfont=dict(size=8),
            name="Hub Nodes",
            hovertext=[
                f"{row['id']}: {row['label']} "
                f"({row['connections']} connections)"
                for _, row in hub_nodes.iterrows()
            ],
        ))

    # Draw other nodes
    other_nodes = nodes_df[nodes_df["type"] != "hub"]
    if len(other_nodes) > 0:
        fig_map.add_trace(go.Scattergeo(
            lon=other_nodes["lon"],
            lat=other_nodes["lat"],
            mode="markers",
            marker=dict(size=4, color="gray",
                        symbol="circle"),
            name="Junctions/Terminals",
            hovertext=[
                f"{row['id']}: {row['label']}"
                for _, row in other_nodes.iterrows()
            ],
        ))

    fig_map.update_geos(
        scope="usa",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        showstates=True,
        statecolor="rgb(200, 200, 200)",
        showlakes=True,
        lakecolor="rgb(220, 230, 240)",
    )
    fig_map.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Operating Gas Pipelines — United States"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Network stats
    col1, col2, col3, col4 = st.columns(4)
    baseline = twin.baseline_kpis
    col1.metric("Total Nodes", baseline["n_nodes"])
    col2.metric("Total Edges", baseline["n_edges"])
    col3.metric("Hub Nodes", len(hub_nodes))
    col4.metric("Connected Components", baseline["n_components"])

# ── TAB 2: STRESS RESULTS ──
with tab2:
    if run_test and G is not None:
        st.subheader(f"Stress Test: {disruption_type.title()} "
                     f"(Severity {severity})")

        with st.spinner("Generating scenarios and running tests..."):
            # Generate scenarios
            scenarios_raw = generate_scenarios(
                G, disruption_type, severity,
                target_region, n_scenarios
            )

            # Convert to disruptions
            disruptions = []
            for s in scenarios_raw:
                d = scenario_to_disruption(
                    s, nodes_df, edges_df, region_assignments
                )
                disruptions.append(d)

            # Run stress tests
            results, summary = run_batch_test(twin, disruptions)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resilience Score",
                     f"{summary['resilience_score']}/100")
        col2.metric("Avg Capacity Retained",
                     f"{summary['avg_capacity_retention']}%")
        col3.metric("Worst Case",
                     f"{summary['min_capacity_retention']}%")
        col4.metric("Max Fragmentation",
                     f"{summary['max_components']} components")

        # Chart 1: Capacity retention distribution
        results_df = pd.DataFrame(results)
        fig_hist = go.Figure(go.Histogram(
            x=results_df["capacity_retention_pct"],
            nbinsx=20,
            marker_color="steelblue"
        ))
        fig_hist.update_layout(
            title="Capacity Retention Distribution",
            xaxis_title="Capacity Retained (%)",
            yaxis_title="Number of Scenarios"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Chart 2: Vulnerability ranking
        st.subheader("Node Vulnerability Ranking")
        with st.spinner("Analyzing individual node vulnerability..."):
            ranked = find_vulnerable_nodes(twin)

        top_10 = ranked[:10]
        vuln_df = pd.DataFrame([
            {
                "Node": node_id,
                "Impact Score": info["impact_score"],
                "Capacity Loss (%)": info["capacity_loss_pct"],
                "Fragmentation": info["fragmentation"],
                "Degree": info["degree"],
                "Type": info["node_type"],
            }
            for node_id, info in top_10
        ])

        fig_vuln = go.Figure(go.Bar(
            x=vuln_df["Impact Score"],
            y=vuln_df["Node"],
            orientation="h",
            marker_color="indianred",
            text=vuln_df["Impact Score"],
            textposition="auto"
        ))
        fig_vuln.update_layout(
            title="Top 10 Most Vulnerable Nodes",
            xaxis_title="Impact Score",
            yaxis_title="Node ID",
            yaxis=dict(autorange="reversed"),
            height=400
        )
        st.plotly_chart(fig_vuln, use_container_width=True)

        st.dataframe(vuln_df, use_container_width=True)

    else:
        st.info("Configure parameters in the sidebar and click "
                "'Run Stress Test' to see results.")

# ── TAB 3: OWNER ANALYSIS ──
with tab3:
    st.subheader("Pipeline Ownership Concentration")

    owner_cap = (edges_df.groupby("owner")["capacity_bcm_y"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(15))

    if len(owner_cap) > 0:
        fig_owner = go.Figure(go.Bar(
            x=owner_cap.values,
            y=owner_cap.index,
            orientation="h",
            marker_color="teal"
        ))
        fig_owner.update_layout(
            title="Top 15 Pipeline Owners by Total Capacity",
            xaxis_title="Total Capacity (BCM/year)",
            yaxis_title="Owner",
            yaxis=dict(autorange="reversed"),
            height=500
        )
        st.plotly_chart(fig_owner, use_container_width=True)

        # Calculate concentration
        total = edges_df["capacity_bcm_y"].sum()
        if total > 0:
            top_owner = owner_cap.index[0]
            top_share = owner_cap.values[0] / total * 100
            st.warning(
                f"⚠️ **Concentration Risk:** {top_owner} controls "
                f"{top_share:.1f}% of total pipeline capacity. "
                f"Operational issues at this company could affect "
                f"a significant portion of the network."
            )
    else:
        st.info("No owner data available in the dataset.")

# ── ABOUT SECTION ──
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    **Data Source:** GEM-GGIT Global Gas Pipelines Dataset (2025)
    — Global Energy Monitor

    **Methodology:**
    1. Real pipeline data converted to a network graph
       (nodes = hubs/terminals, edges = pipeline segments)
    2. Conditional GAN trained on patterns from 15+ real
       disruption events (hurricanes, cyberattacks, sanctions)
    3. GAN generates novel disruption scenarios conditioned
       on type, severity, and target region
    4. Digital twin simulates each scenario and measures
       network connectivity and capacity impact

    **Disruption Types:**
    - **Hurricane:** Geographically clustered impact on coastal regions
    - **Sabotage:** Targeted shutdown of specific pipeline corridors
    - **Sanctions:** Sustained capacity reduction from geopolitical events
    - **Demand Surge:** Increased demand exceeding pipeline capacity
    - **Multi-Factor:** Combination of multiple disruption types
    """)
```

```
WHAT CAN GO WRONG WITH THE DASHBOARD:

PROBLEM: "ModuleNotFoundError" when running streamlit
FIX:     Run from project root:
         cd gas-pipeline-stress-tester
         streamlit run app.py

PROBLEM: Map shows no pipelines (empty map)
CAUSE 1: Geometry data is missing or could not be parsed
FIX:     Check edges_df["geometry"].head() — if all NaN or
         empty strings, the geometry was not saved.
         Go back to data_processor and verify geometry is saved.
CAUSE 2: json.loads fails silently due to try/except
FIX:     Temporarily remove the try/except to see errors:
         geom = json.loads(str(geom_str))
         This will crash and show you the actual error.

PROBLEM: Map shows pipelines in the wrong location (ocean, etc.)
CAUSE:   Lat/lon swapped
FIX:     In the map drawing code, ensure:
         lons = [p[0] for p in segment]  ← longitude first
         lats = [p[1] for p in segment]  ← latitude second
         GeoJSON is [lon, lat] and Plotly Scattergeo uses
         separate lon= and lat= parameters.

PROBLEM: Streamlit runs but is very slow (takes 30+ seconds to load)
CAUSE 1: Loading large GeoJSON file every time
FIX:     Use @st.cache_resource and @st.cache_data decorators
         (already done above). These cache expensive operations.
CAUSE 2: geometry column in edges.csv is very large
FIX:     Simplify geometries by keeping every Nth point:
         simplified = segment[::5]  # Keep every 5th point
         This reduces rendering load dramatically.

PROBLEM: "st.cache_data" does not work with underscore prefix
FIX:     When passing non-serializable objects (like DataFrames)
         to cached functions, prefix the parameter with underscore:
         def get_region_assignments(_nodes_df):
         The underscore tells Streamlit not to hash that parameter.

PROBLEM: Stress test button does nothing
CAUSE:   The run_test variable is False because button was not clicked
         Streamlit reruns the entire script on every interaction.
FIX:     This is already handled with the if run_test: block.
         Make sure the button and the results display are in the
         same tab.

PROBLEM: find_vulnerable_nodes is too slow in the dashboard
CAUSE:   It tests every single node, which takes minutes
FIX:     Only test the top 20 nodes by degree:
         top_by_degree = sorted(twin.graph.degree(),
                               key=lambda x: x[1], reverse=True)[:20]
         Then only test those 20 nodes.
         Or cache the results:
         @st.cache_data
         def cached_vulnerability_analysis():
             ...
```

**Run the dashboard:**
```bash
cd gas-pipeline-stress-tester
streamlit run app.py
```

This opens a browser at `http://localhost:8501`.

**Done when:** All three tabs display correctly, the map shows real pipelines, stress test produces results, and owner analysis shows concentration data.

---

### PHASE 9: DEPLOYMENT

**Step 9.1 — Prepare for GitHub**

Create a `.gitignore` file:
```
__pycache__/
*.pyc
.DS_Store
venv/
.env
```

**Step 9.2 — Push to GitHub**

```bash
git init
git add .
git commit -m "Gas Pipeline Stress Tester - Complete"
git remote add origin https://github.com/YOUR_USERNAME/gas-pipeline-stress-tester.git
git push -u origin main
```

```
PROBLEM: GeoJSON file is too large for GitHub (>100MB)
FIX:     Add it to .gitignore and provide download
         instructions in README.
         Option B: Use Git LFS (Large File Storage):
         git lfs install
         git lfs track "*.geojson"
         git add .gitattributes
         Option C: Only push the processed CSVs (much smaller)
         and add the GeoJSON to .gitignore.
         The processed CSVs are all you need for the app to run.
```

**Step 9.3 — Deploy to Streamlit Cloud**

1. Go to share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path to `app.py`
6. Click "Deploy"

```
PROBLEM: Deployment fails with "File too large"
FIX:     Remove the GeoJSON from the repo.
         The app only needs nodes.csv, edges.csv,
         and generator.pth. The GeoJSON was only needed
         for the initial data processing.

PROBLEM: Deployment fails with "ModuleNotFoundError"
FIX:     Make sure requirements.txt is in the repo root.
         Add any missing libraries.

PROBLEM: App crashes on Streamlit Cloud with MemoryError
FIX:     Reduce data size. Simplify geometries.
         Use edges.csv without the geometry column
         and draw simple lines between node coordinates
         instead of actual pipeline routes.

PROBLEM: torch is too large for Streamlit Cloud
FIX:     Use: torch --index-url https://download.pytorch.org/whl/cpu
         in requirements.txt. Or replace with:
         --extra-index-url https://download.pytorch.org/whl/cpu
         torch
```

---

## 5. Conclusion

This project demonstrates a complete framework for stress testing critical gas pipeline infrastructure using real-world data and generative AI. The system takes the GEM-GGIT dataset containing actual pipeline routes, capacities, and ownership information for operating gas pipelines, and converts it into a graph-based digital twin through geographic clustering of pipeline endpoints.

The data processing pipeline addresses the fundamental challenge that pipeline datasets provide edges (pipeline segments) but not explicit nodes (infrastructure hubs). By clustering nearby pipeline endpoints using a distance-based algorithm, the system automatically identifies hub nodes where multiple pipelines converge, which are the locations most likely to be single points of failure.

The Conditional GAN is trained on disruption patterns derived from 18 documented real-world events spanning natural disasters (Hurricane Katrina, Winter Storm Uri), cyberattacks (Colonial Pipeline ransomware), and geopolitical disruptions (Russia-Europe gas cutoff). Rather than randomly sampling failure scenarios, the GAN learns the correlated, geographically clustered nature of real disruptions. A hurricane does not knock out one random node — it simultaneously disables every pipeline in a coastal region while leaving inland infrastructure largely intact. The GAN captures and reproduces these spatial correlation patterns.

The stress testing engine systematically evaluates network resilience by measuring three complementary metrics: network connectivity fragmentation (how many disconnected pieces the network breaks into), total capacity retention (what percentage of pipeline capacity remains operational), and individual node vulnerability (which specific hubs, if disabled, would cause the most damage). The combination of these metrics provides a resilience score that quantifies overall network robustness.

The interactive dashboard makes these analyses accessible through real pipeline route visualization on geographic maps, configurable disruption scenario generation, vulnerability rankings, and corporate ownership concentration analysis. The ownership analysis reveals an often-overlooked risk dimension: if a small number of companies control a large share of critical pipeline capacity, operational failures, financial distress, or regulatory action affecting a single company could have outsized network-wide consequences.

Key findings for the US gas pipeline network typically include the identification of 5-8 critical hub nodes (often in the Gulf Coast region around Louisiana and Texas) whose failure would fragment the network into multiple disconnected components, the observation that hurricanes targeting the Gulf Coast produce the most severe disruption scenarios due to the concentration of pipeline infrastructure in that region, and the revelation that corporate ownership concentration creates additional systemic risk beyond what pure network topology analysis would suggest.

The framework is extensible to any region covered by the GEM-GGIT dataset (Europe, Middle East, Asia) by changing the country filter and regional definitions, making it a general-purpose tool for energy infrastructure resilience assessment.
