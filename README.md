# Black-Scholes-Visualizer
Visualizing Black-Scholes-Merton Algorithm in 3D surface plots


# Black-Scholes Option Pricing & Greeks Visualizer (INR)

An interactive app to compute and visualize European option prices and Greeks using the Black-Scholes model, with all inputs/outputs in INR.

---

## Features

* European call and put pricing with all major Greeks (Delta, Gamma, Vega, Theta, Rho)
* **2D heatmaps:** Option price vs any two inputs
* **3D surface plots:** Any price/Greek vs any two parameters
* **1D line plots:** Any Greek vs a chosen input
* Sidebar controls for all parameters
* Instant updates on input change
* Robust input validation

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Go to the local URL in your browser.

---

## Parameters

* **Spot Price (S)**: ₹
* **Strike Price (K)**: ₹
* **Time to Maturity (T)**: years
* **Volatility (σ)**: annualized, decimal
* **Interest Rate (r)**: annualized, decimal

---

## License

MIT

---

For educational and reference purposes only. Not financial advice.
