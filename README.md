# ◈ PriceOracle — AI Dynamic Pricing & Profit Intelligence

> **What price should I set to maximise profit — right now?**  
> PriceOracle answers that question with machine learning demand prediction, price simulation, and AI-generated strategic recommendations — built for Amazon India fashion retail.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Table of Contents

- [The Business Problem](#the-business-problem)
- [Key Insights from the Data](#key-insights-from-the-data)
- [ML Model Performance](#ml-model-performance)
- [What Drives Demand — Feature Importance](#what-drives-demand--feature-importance)
- [The Optimization Engine](#the-optimization-engine)
- [App Features](#app-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [5 Business Recommendations](#5-business-recommendations)

---

## The Business Problem

Pricing decisions on Amazon India are made blindly — with significant financial consequences at scale.

| Problem | Impact |
|---------|--------|
| **Gut-feel pricing** | No system to find the profit-maximising sweet spot — too low loses margin, too high loses customers |
| **Competitor blind spots** | Sellers are last to know when a competitor drops prices, losing festive-season sales before reacting |
| **Missed festive revenue** | October & November deliver **3× the revenue** of January — sellers who under-price during this window leave money on the table |

**PriceOracle solves this with a four-stage pipeline:**

```
Predict Demand  →  Simulate 60 Price Points  →  Optimise Profit  →  Explain with AI
```

---

## Key Insights from the Data

The model was trained on **113,815 Amazon India fashion orders** across the full year 2022, covering 37 states/UTs and 7 product categories.

### Seasonal Revenue Patterns

The most striking pattern in the data is the **festive season surge**:

- October revenue alone is **3× higher** than January
- Average demand during September–November: **6.5 units/cycle**
- Average demand during January–August: **5.4 units/cycle**
- Demand elasticity is measurably lower during Oct–Nov — customers are less price-sensitive and more purchase-intent driven

> **Insight:** Sellers can raise prices by 10–15% in October and November without a proportional loss in volume. At ₹599, a Kurta can become ₹699 during the festive window with no meaningful demand drop.

---

### Category Performance

| Category | Records | Avg Price | Share of Orders |
|----------|---------|-----------|-----------------|
| Kurta | 48,717 | ₹846 | 43% |
| Set | 30,752 | ₹1,298 | 27% |
| Top | 11,454 | ₹597 | 10% |
| Western Dress | 9,088 | ₹1,545 | 8% |
| Skirt | 8,005 | ₹549 | 7% |
| Ethnic Dress | 3,532 | ₹954 | 3% |
| Blouse | 2,267 | ₹400 | 2% |

**Key findings:**

- **Kurta** is the #1 revenue driver — volume dominates. Price optimisation here has the largest absolute impact.
- **Western Dress** carries the highest average price at ₹1,545. Strong demand + premium positioning = do not discount this category.
- **Skirt and Blouse** are underpenetrated growth categories. Both are priced low relative to demand potential — they represent pricing upside.
- **Set** is the second-highest revenue category by volume but priced at ₹1,298 average — the most profitable cohort per unit sold.

---

### Regional Revenue Distribution

| Region | Total Revenue | Tier |
|--------|--------------|------|
| Maharashtra | ₹80.2M | Metro |
| Karnataka | ₹53.8M | Metro |
| Telangana | ₹40.1M | Metro |
| Tamil Nadu | ₹36.1M | Metro |
| Uttar Pradesh | ₹31.2M | Tier 1 |
| Delhi | ₹26.2M | Metro |

**Maharashtra + Karnataka alone account for 27% of total revenue.** Metro regions consistently show higher average order values and lower price sensitivity — presenting a clear case for regional price differentiation.

> **Insight:** A ₹50–100 metro premium on top-performing categories in Maharashtra, Karnataka, and Telangana is supportable based on the purchasing power signals in this dataset.

---

### Competitor Pricing Analysis

| Metric | Value |
|--------|-------|
| Average competitor price | ₹991 |
| Average our price | ₹973 |
| Price gap | **₹18 cheaper on average** |

The current pricing is systemically below-market. This represents unrealised margin — especially in Kurta and Set categories where the gap is widest. 

**Feature correlation highlights:**

| Pair | Correlation | Interpretation |
|------|-------------|----------------|
| Price ↔ Demand | **−0.78** | Strong inverse — price is the primary demand lever |
| Price ↔ Competitor | **+0.97** | Prices move closely together across the market |
| Price ↔ Cost | **+0.99** | Cost is the primary price anchor — opportunity exists above it |
| Festive ↔ Demand | **+0.70** | Festive season meaningfully lifts demand |
| Revenue ↔ Profit | **+0.94** | Volume and profit are well-aligned |

---

## ML Model Performance

Three models were trained and evaluated on demand prediction (units sold):

| Model | R² Score | MAE | MAPE |
|-------|----------|-----|------|
| Random Forest | 0.967 | 0.597 | 12.2% |
| **Gradient Boosting** | **0.978** | **0.511** | **10.7%** |
| XGBoost | 0.977 | 0.511 | 10.6% |

**Gradient Boosting was selected** as the production model (marginally better R² and interpretability).

### What These Numbers Mean

- **R² = 0.978** — the model explains 97.8% of the variance in demand. It knows what drives volume.
- **MAE = 0.511** — predictions are off by less than half a unit on average. For a demand range of 1–20 units, this is operationally precise.
- **MAPE = 10.7%** — average percentage error of ~10.7% across all products, regions, and price points.

> In plain terms: before a product launches at a new price, the model predicts how many units will sell — with ~10% error. That's enough precision to make confident pricing decisions.

---

## What Drives Demand — Feature Importance

From the Gradient Boosting feature importance analysis:

| Rank | Feature | Importance | Business Meaning |
|------|---------|------------|-----------------|
| #1 | `price` | **84%** | The single biggest lever — price dwarfs everything else |
| #2 | `price_tier_enc` | **7%** | Budget / Mid-range / Premium positioning matters to buyers |
| #3 | `category_enc` | **5%** | Kurtas and Sets behave very differently from Blouses |
| #4 | `cost_price` | ~1.5% | Indirectly signals product quality to the model |
| #5 | `is_festive_season` | ~1% | Measurable demand lift in Sep–Nov window |

**All other features** (competitor price, discount, region, day of week) individually contribute less than 1% — but they compound meaningfully in the profit optimisation sweep.

> **The critical implication:** If price is 84% of the demand signal, then pricing is not a secondary concern — it is the primary product management decision. Every repricing event without simulation is leaving money on the table.

---

## The Optimization Engine

The optimizer runs a **60-point price sweep** between configurable floor and ceiling bounds, predicts demand at each price point, calculates expected profit, and returns the price that maximises `(Price − Cost) × Predicted Demand`.

### Real Example — Kurta at ₹799

| Scenario | Price | Predicted Demand | Profit |
|----------|-------|-----------------|--------|
| Current | ₹799 | 8.1 units | ₹2,593 |
| **Optimal** | **₹1,043** | **6.9 units** | **₹3,577** |
| **Improvement** | | | **+38%** |

Even with lower volume at the higher price, the per-unit margin increase more than compensates — a classic price-elasticity optimisation outcome.

### How the 4-Step Process Works

```
Step 1: Feature Encoding
  → Category, size, region, date, price, cost, competitor price, discount
  → Derived: price_diff, price_ratio, profit_margin_pct, price_tier, region_tier, is_festive_season

Step 2: Demand Prediction
  → ML model predicts units sold at 60 price points across the configured range

Step 3: Profit Calculation
  → (Price − Cost) × Predicted Demand for every simulated price

Step 4: Optimal Price Extraction
  → argmax(profit curve) → recommended price + profit improvement %
```

---

## App Features

### Price Optimizer Tab
- Input product details (category, size, region, date) and pricing (own price, cost, competitor, discount)
- Live gross margin validation with visual alerts
- KPI dashboard: predicted demand, current profit, competitor gap, festive status
- Profit and demand curves with optimal price marked
- Side-by-side current vs optimal comparison bar charts
- Business insight chips: elasticity estimate, regional tier, competitor positioning, festive status, margin at optimal

### AI Recommendation Section *(dedicated, on-demand)*
- Separate panel with explicit **Generate AI Recommendation** button — does not auto-fire on every optimizer run
- Powered by **Groq LLaMA 3.1 8B Instant** (or rule-based fallback when API key is absent)
- Produces 3–4 sentence professional recommendation: direction, rationale, and one actionable tactic
- Independent session state — AI text persists across optimizer re-runs until cleared
- Clear button to reset and regenerate without re-running the optimizer

### Scenario Lab Tab
- Interactive what-if heatmap: profit across competitor price × discount grid
- Line chart: profit vs competitor price at any fixed discount level
- Configurable competitor price range and discount range

### Batch Analysis Tab
- Generate and optimise 5–50 products simultaneously
- Summary KPIs: average uplift %, products priced up/down, total profit gain
- Sorted improvement bar chart
- Colour-gradient results table
- CSV export

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **App framework** | Streamlit 1.32+ |
| **ML / modelling** | scikit-learn (Gradient Boosting, Random Forest), XGBoost |
| **Data processing** | Pandas, NumPy |
| **Visualisation** | Plotly (line, bar, heatmap, subplots) |
| **Model persistence** | Joblib (`.pkl` serialisation) |
| **AI recommendations** | Groq API — LLaMA 3.1 8B Instant |
| **Fonts** | DM Sans, DM Mono, Syne (Google Fonts) |
| **Environment config** | python-dotenv |
| **Data source** | Amazon India fashion orders — 113,815 records, full year 2022 |

---

## Project Structure

```
PriceOracle/
│
├── priceoracle_app.py          # Main Streamlit application
│
├── model_artifacts/
│   ├── demand_model.pkl        # Trained Gradient Boosting model
│   └── metadata.json           # Training metadata & feature schema
│
├── PriceOracle_ML_Model.ipynb  # Full training notebook:
│                               #   EDA → feature engineering →
│                               #   model training → evaluation →
│                               #   optimization logic
│
├── .env                        # Environment variables (not committed)
│   └── GROQ_API_KEY=gsk_...
│
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/PriceOracle.git
cd PriceOracle
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly requests python-dotenv
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). Without this key, the app runs in **rule-based fallback mode** — all pricing features still work, only the AI narrative recommendation falls back to a heuristic.

### 4. Add the trained model *(optional)*

Place your trained model at `model_artifacts/demand_model.pkl`. Without it, the app runs in **simulation mode** using the built-in heuristic demand function — fully functional for exploration and demonstration.

### 5. Run the app

```bash
streamlit run priceoracle_app.py
```

---

## Configuration

All optimizer parameters are configurable from the sidebar at runtime:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Price floor | 70% | 50–90% | Lower bound of price sweep (% of current price) |
| Price ceiling | 135% | 110–200% | Upper bound of price sweep (% of current price) |
| Simulation steps | 60 | 20–100 | Number of price points evaluated (higher = smoother curves) |

The price floor also enforces a hard constraint of `cost_price × 1.05` — the optimizer will never recommend a price below a 5% margin over cost.

---

## 5 Business Recommendations

Derived from 113,815 orders of Amazon India data:

**01 · HIGH IMPACT — Raise prices 10–15% in October and November**  
Festive demand is 3× normal. Customers are less price-sensitive. A ₹599 Kurta can become ₹699 without losing volume — the model demonstrates this at elasticity ≈ 0.7 during the festive window.

**02 · QUICK WIN — Fix underpriced products**  
Average price (₹973) is ₹18 below competitor average (₹991). For Kurtas and Sets, a 5–10% price increase recaptures margin with minimal demand impact given the −0.78 correlation between price and demand.

**03 · MARGIN PROTECT — Do not discount Western Dress**  
At an average of ₹1,545 with strong demand, Western Dress is the premium positioning play. Discounting this category destroys margin without a proportional volume uplift — the price tier signal in the model shows premium buyers are not discount-driven.

**04 · GROWTH — Regional price differentiation**  
Maharashtra and Karnataka alone drive 27% of total revenue at above-average price points. A ₹50–100 metro premium in top-5 regions is supportable and adds directly to margin with no additional cost.

**05 · SYSTEMATIC — Use PriceOracle before every repricing decision**  
Before changing any price, run it through the optimizer. The simulation takes under 5 seconds and shows expected demand, profit, and improvement — replacing intuition with a repeatable, data-driven process.

---

## Dashboard KPIs (Full Year 2022)

| Metric | Value |
|--------|-------|
| Total Revenue Analysed | ₹475M |
| Total Quantity | 678K units |
| Total Profit | ₹178M |
| Overall Profit Margin | 38% |
| Average Profit Improvement (optimised) | +38% |
| Model R² (Gradient Boosting) | 0.978 |

---

## License

MIT License — see `LICENSE` for details.

---

*Built with Python · Gradient Boosting · Streamlit · Groq AI · Plotly*  
*PriceOracle © 2026*
