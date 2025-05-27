import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt # Keep for heatmaps for now
import seaborn as sns # Keep for heatmaps for now

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing (INR)",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.metric-container {
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    height: 100px; margin-bottom: 10px;
}
.metric-call { background-color: #e6ffed; border: 1px solid #5cb85c; }
.metric-put { background-color: #ffebee; border: 1px solid #d9534f; }
.metric-greek { background-color: #e3f2fd; border: 1px solid #90caf9; }
.metric-value { font-size: 1.8rem; font-weight: bold; color: #333; margin-top: 5px; }
.metric-label { font-size: 1.0rem; font-weight: 500; color: #555; text-align: center; }
div[data-baseweb="tab-list"] { justify-content: center; }
div[data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; margin-right: 4px; }
div[data-baseweb="tab"][aria-selected="true"] { background-color: #007bff; color: white; }
.stPlotlyChart { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# --- Parameter Definitions ---
PARAM_SPOT = "Spot Price (S)"
PARAM_STRIKE = "Strike Price (K)"
PARAM_TIME = "Time to Maturity (T)"
PARAM_VOL = "Volatility (σ)"
PARAM_RATE = "Interest Rate (r)"
ALL_INPUT_PARAMS = [PARAM_SPOT, PARAM_STRIKE, PARAM_TIME, PARAM_VOL, PARAM_RATE]

PARAM_TO_KEY_MAP = {
    PARAM_SPOT: 'current_price', PARAM_STRIKE: 'strike',
    PARAM_TIME: 'time_to_maturity', PARAM_VOL: 'volatility',
    PARAM_RATE: 'interest_rate'
}

# --- Output Definitions (for Z-axis in 3D plots and Y-axis in 2D Greek plots) ---
# Structure: Display Name: {'key': bs_attribute_name, 'scaler': for_display, 'unit': display_unit_string}
OUTPUT_PARAM_DETAILS = {
    "Call Price":       {'key': 'call_price',   'scaler': 1,     'unit': '₹'},
    "Put Price":        {'key': 'put_price',    'scaler': 1,     'unit': '₹'},
    "Call Delta":       {'key': 'call_delta',   'scaler': 1,     'unit': ''},
    "Put Delta":        {'key': 'put_delta',    'scaler': 1,     'unit': ''},
    "Gamma":            {'key': 'gamma',        'scaler': 1,     'unit': ''},
    "Vega":             {'key': 'vega',         'scaler': 0.01,  'unit': '₹ per 1% vol pt.'},
    "Call Theta":       {'key': 'call_theta',   'scaler': 1/365, 'unit': '₹ per day'},
    "Put Theta":        {'key': 'put_theta',    'scaler': 1/365, 'unit': '₹ per day'},
    "Call Rho":         {'key': 'call_rho',     'scaler': 0.01,  'unit': '₹ per 1% rate pt.'},
    "Put Rho":          {'key': 'put_rho',      'scaler': 0.01,  'unit': '₹ per 1% rate pt.'},
}
PLOTTABLE_OUTPUTS_DISPLAY_NAMES = list(OUTPUT_PARAM_DETAILS.keys())
GREEKS_DISPLAY_NAMES = [name for name in PLOTTABLE_OUTPUTS_DISPLAY_NAMES if "Price" not in name]


class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, current_price: float, volatility: float, interest_rate: float):
        if not (isinstance(time_to_maturity, (int, float)) and time_to_maturity > 0): raise ValueError("Time to maturity must be > 0.")
        if not (isinstance(strike, (int, float)) and strike > 0): raise ValueError("Strike price must be > 0.")
        if not (isinstance(current_price, (int, float)) and current_price > 0): raise ValueError("Current price must be > 0.")
        if not (isinstance(volatility, (int, float)) and volatility > 0): raise ValueError("Volatility must be > 0.")
        if not isinstance(interest_rate, (int, float)): raise ValueError("Interest rate must be a number.")

        self.time_to_maturity, self.strike, self.current_price = time_to_maturity, strike, current_price
        self.volatility, self.interest_rate = volatility, interest_rate
        # Initialize attributes
        attrs_to_init = ['d1', 'd2', 'call_price', 'put_price', 'call_delta', 'put_delta', 
                         'gamma', 'vega', 'call_theta', 'put_theta', 'call_rho', 'put_rho']
        for attr in attrs_to_init: setattr(self, attr, 0.0)

    def _calculate_d1_d2(self):
        sqrt_T_local = sqrt(self.time_to_maturity)
        denominator = self.volatility * sqrt_T_local
        if denominator == 0:
            self.d1 = np.nan if self.current_price == self.strike else np.inf * np.sign(log(self.current_price/self.strike)) if self.current_price > 0 and self.strike > 0 else np.nan
            self.d2 = np.nan
            return
        self.d1 = (log(self.current_price / self.strike) + (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / denominator
        self.d2 = self.d1 - denominator

    def run_calculations(self):
        self._calculate_d1_d2()
        if np.isnan(self.d1) or np.isnan(self.d2):
            for attr_key in OUTPUT_PARAM_DETAILS: # Set all calculable outputs to NaN
                setattr(self, OUTPUT_PARAM_DETAILS[attr_key]['key'], np.nan)
            return

        S, K, T, sigma, r = self.current_price, self.strike, self.time_to_maturity, self.volatility, self.interest_rate
        sqrt_T, exp_rt = sqrt(T), exp(-r * T)
        K_exp_rt = K * exp_rt
        N_d1, N_d2, N_neg_d1, N_neg_d2 = norm.cdf(self.d1), norm.cdf(self.d2), norm.cdf(-self.d1), norm.cdf(-self.d2)
        N_prime_d1 = norm.pdf(self.d1)

        self.call_price, self.put_price = S * N_d1 - K_exp_rt * N_d2, K_exp_rt * N_neg_d2 - S * N_neg_d1
        self.call_delta, self.put_delta = N_d1, N_d1 - 1
        
        denom_gamma = S * sigma * sqrt_T
        self.gamma = N_prime_d1 / denom_gamma if denom_gamma != 0 else np.nan
        self.vega = S * N_prime_d1 * sqrt_T

        common_theta_num = S * N_prime_d1 * sigma
        common_theta_den = 2 * sqrt_T
        common_theta_term = (common_theta_num / common_theta_den) if common_theta_den != 0 else np.nan
        
        self.call_theta = -common_theta_term - r * K_exp_rt * N_d2
        self.put_theta = -common_theta_term + r * K_exp_rt * N_neg_d2
        self.call_rho, self.put_rho = T * K_exp_rt * N_d2, -T * K_exp_rt * N_neg_d2

# --- Sidebar for User Inputs (Base Values) ---
with st.sidebar:
    st.title("₹ Black-Scholes Inputs")
    st.markdown("---")
    st.subheader("Core Option Parameters (Base Values)")
    sb_current_price = st.number_input(PARAM_SPOT.split('(')[0].strip()+ " (₹)", min_value=0.01, value=10000.0, step=100.0)
    sb_strike = st.number_input(PARAM_STRIKE.split('(')[0].strip() + " (₹)", min_value=0.01, value=10000.0, step=100.0)
    sb_time_to_maturity = st.number_input(PARAM_TIME.split('(')[0].strip()+ " (Years)", min_value=0.001, value=1.0, step=0.05, format="%.3f")
    sb_volatility_pct = st.slider(PARAM_VOL.split('(')[0].strip()+ " (%)", min_value=1, max_value=100, value=20, step=1)
    sb_interest_rate_pct = st.slider(PARAM_RATE.split('(')[0].strip()+ " (%)", min_value=-5, max_value=20, value=5, step=1)

    # Store base values for sensitivity analysis
    base_S, base_K, base_T = sb_current_price, sb_strike, sb_time_to_maturity
    base_vol, base_r = sb_volatility_pct / 100.0, sb_interest_rate_pct / 100.0

    st.markdown("---")
    st.subheader("Global Sensitivity Plot Settings")
    num_sensitivity_points = st.slider("Number of Points for Sensitivity Axes", 10, 50, 25, step=5) # Reduced max for performance

# --- Helper Functions for UI and Data Generation ---
def get_input_param_details(param_name):
    details = {
        PARAM_SPOT: {'base': base_S, 'label': "Spot Price (₹)", 'min': 0.01, 'step': 100.0, 'format': "%.2f"},
        PARAM_STRIKE: {'base': base_K, 'label': "Strike Price (₹)", 'min': 0.01, 'step': 100.0, 'format': "%.2f"},
        PARAM_TIME: {'base': base_T, 'label': "Time to Maturity (Years)", 'min': 0.001, 'step': 0.01, 'format': "%.3f"},
        PARAM_VOL: {'base': base_vol, 'label': "Volatility (decimal)", 'min': 0.001, 'step': 0.01, 'format': "%.4f"},
        PARAM_RATE: {'base': base_r, 'label': "Interest Rate (decimal)", 'min': -1.0, 'step': 0.005, 'format': "%.4f"}
    }
    return details[param_name]

def get_default_range_for_input_param(param_name, base_value):
    factor = 0.2 # 20% range
    if param_name == PARAM_SPOT or param_name == PARAM_STRIKE: return base_value * (1-factor), base_value * (1+factor)
    if param_name == PARAM_TIME: return max(0.001, base_value * (1-factor*2)), base_value * (1+factor*2) # Wider for time
    if param_name == PARAM_VOL: return max(0.001, base_value * (1-factor*1.5)), base_value * (1+factor*1.5)
    if param_name == PARAM_RATE: return base_value - 0.02, base_value + 0.02 # Absolute for rates
    return base_value * (1-factor), base_value * (1+factor)

def generate_input_param_range_ui(param_name_key_prefix, default_param_name, num_points, col_obj=st):
    param_details = get_input_param_details(default_param_name)
    def_min, def_max = get_default_range_for_input_param(default_param_name, param_details['base'])
    
    min_val = col_obj.number_input(f"Min {param_details['label']}", value=float(def_min), 
                                   min_value=float(param_details['min']) if param_details['min'] is not None else None,
                                   step=float(param_details['step']), format=param_details['format'], key=f"{param_name_key_prefix}_min")
    max_val = col_obj.number_input(f"Max {param_details['label']}", value=float(def_max), 
                                   min_value=float(min_val) + float(param_details['step']),
                                   step=float(param_details['step']), format=param_details['format'], key=f"{param_name_key_prefix}_max")
    
    created_range = np.linspace(min_val, max_val, num_points)
    if param_details['min'] is not None: created_range = np.maximum(created_range, param_details['min'])
    return created_range

@st.cache_data
def generate_surface_data(_bs_base_params_dict, x_input_param_name, x_range, y_input_param_name, y_range, z_output_display_name):
    z_output_details = OUTPUT_PARAM_DETAILS[z_output_display_name]
    z_values = np.zeros((len(y_range), len(x_range)))
    x_key, y_key = PARAM_TO_KEY_MAP[x_input_param_name], PARAM_TO_KEY_MAP[y_input_param_name]
    z_attr_key, z_scaler = z_output_details['key'], z_output_details['scaler']

    for i, y_val in enumerate(y_range):
        for j, x_val in enumerate(x_range):
            temp_params = _bs_base_params_dict.copy()
            temp_params[x_key], temp_params[y_key] = x_val, y_val
            try:
                bs = BlackScholes(**temp_params)
                bs.run_calculations()
                raw_z_val = getattr(bs, z_attr_key)
                z_values[i, j] = raw_z_val * z_scaler if not np.isnan(raw_z_val) else np.nan
            except (ValueError, AttributeError):
                z_values[i, j] = np.nan
    return z_values

@st.cache_data
def generate_1d_plot_data(_bs_base_params_dict, varying_input_param_name, varying_range, target_output_display_name):
    target_output_details = OUTPUT_PARAM_DETAILS[target_output_display_name]
    varying_key = PARAM_TO_KEY_MAP[varying_input_param_name]
    target_attr_key, target_scaler = target_output_details['key'], target_output_details['scaler']
    
    output_values = []
    for val in varying_range:
        temp_params = _bs_base_params_dict.copy()
        temp_params[varying_key] = val
        try:
            bs = BlackScholes(**temp_params)
            bs.run_calculations()
            raw_val = getattr(bs, target_attr_key)
            output_values.append(raw_val * target_scaler if not np.isnan(raw_val) else np.nan)
        except (ValueError, AttributeError):
            output_values.append(np.nan)
    return pd.Series(output_values, index=varying_range)


# --- Main Page ---
st.title("Black-Scholes Option Pricing & Analysis (INR)")
st.markdown("A tool to calculate European option prices, Greeks, and visualize their sensitivity.")
st.markdown("---")

# --- Model Inputs Summary & Initial Calculation ---
st.subheader("Model Inputs Summary (Base Values)")
bs_base_input_values = [f"₹{base_S:,.2f}", f"₹{base_K:,.2f}", f"{base_T:.3f} Years",
                        f"{base_vol*100:.0f}% ({base_vol:.4f})", f"{base_r*100:.1f}% ({base_r:.4f})"]
st.table(pd.DataFrame({"Parameter": [p.split('(')[0].strip() for p in ALL_INPUT_PARAMS], "Value": bs_base_input_values}).set_index("Parameter"))

bs_model = None
calculation_successful = False
try:
    bs_model = BlackScholes(base_T, base_K, base_S, base_vol, base_r)
    bs_model.run_calculations()
    if np.isnan(bs_model.call_price): raise ValueError("Calculation resulted in NaN.")
    calculation_successful = True
except ValueError as e:
    st.error(f"Error in Black-Scholes calculation with base inputs: {e}")

# --- Display Main Metrics ---
st.subheader("Calculated Option Metrics (Based on Base Values)")
# ... (Metrics display remains similar, ensure bs_model attributes are used)
if calculation_successful and bs_model:
    # (Metric display code - slightly condensed for brevity, it's the same logic as before)
    mc1, mc2 = st.columns(2)
    mc1.markdown(f'<div class="metric-container metric-call"><div class="metric-label">CALL Value</div><div class="metric-value">₹{bs_model.call_price:.2f}</div></div>', unsafe_allow_html=True)
    mc2.markdown(f'<div class="metric-container metric-put"><div class="metric-label">PUT Value</div><div class="metric-value">₹{bs_model.put_price:.2f}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Key Option Greeks (Based on Base Values)")
    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Call Delta"]["unit"] if OUTPUT_PARAM_DETAILS["Call Delta"]["unit"] else "Call Delta"}</div><div class="metric-value">{bs_model.call_delta * OUTPUT_PARAM_DETAILS["Call Delta"]["scaler"]:.4f}</div></div>', unsafe_allow_html=True)
    gc1.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Put Delta"]["unit"] if OUTPUT_PARAM_DETAILS["Put Delta"]["unit"] else "Put Delta"}</div><div class="metric-value">{bs_model.put_delta * OUTPUT_PARAM_DETAILS["Put Delta"]["scaler"]:.4f}</div></div>', unsafe_allow_html=True)
    gc2.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Gamma"]["unit"] if OUTPUT_PARAM_DETAILS["Gamma"]["unit"] else "Gamma"}</div><div class="metric-value">{bs_model.gamma * OUTPUT_PARAM_DETAILS["Gamma"]["scaler"]:.4f}</div></div>', unsafe_allow_html=True)
    gc2.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Vega"]["unit"]}</div><div class="metric-value">{bs_model.vega * OUTPUT_PARAM_DETAILS["Vega"]["scaler"]:.2f}</div></div>', unsafe_allow_html=True)
    gc3.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Call Theta"]["unit"]}</div><div class="metric-value">{bs_model.call_theta * OUTPUT_PARAM_DETAILS["Call Theta"]["scaler"]:.2f}</div></div>', unsafe_allow_html=True)
    gc3.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Put Theta"]["unit"]}</div><div class="metric-value">{bs_model.put_theta * OUTPUT_PARAM_DETAILS["Put Theta"]["scaler"]:.2f}</div></div>', unsafe_allow_html=True)
    gc4.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Call Rho"]["unit"]}</div><div class="metric-value">{bs_model.call_rho * OUTPUT_PARAM_DETAILS["Call Rho"]["scaler"]:.2f}</div></div>', unsafe_allow_html=True)
    gc4.markdown(f'<div class="metric-container metric-greek"><div class="metric-label">{OUTPUT_PARAM_DETAILS["Put Rho"]["unit"]}</div><div class="metric-value">{bs_model.put_rho * OUTPUT_PARAM_DETAILS["Put Rho"]["scaler"]:.2f}</div></div>', unsafe_allow_html=True)

else: st.info("Metrics & Greeks will be displayed once base inputs are valid.")
st.markdown("---")

# --- Sensitivity Analysis Tabs ---
if calculation_successful:
    base_params_for_sensitivity_dict = {
        'current_price': base_S, 'strike': base_K, 'time_to_maturity': base_T,
        'volatility': base_vol, 'interest_rate': base_r
    }
    tab_heatmap, tab_3d, tab_2d_greek = st.tabs(["📈 Price Heatmaps (2D)", "🧊 Surface Plots (3D)", "📊 Greek Analysis (2D)"])

    with tab_heatmap:
        st.header("Option Price Heatmaps")
        st.markdown("Shows Call/Put prices against two varying input parameters. Others fixed from sidebar.")
        ht_c1, ht_c2 = st.columns(2)
        x_param_ht = ht_c1.selectbox("X-axis Parameter:", ALL_INPUT_PARAMS, index=0, key="ht_x_param")
        y_param_ht = ht_c2.selectbox("Y-axis Parameter:", ALL_INPUT_PARAMS, index=3, key="ht_y_param")

        if x_param_ht == y_param_ht: st.error("X and Y axes must be different.")
        else:
            ht_rc1, ht_rc2 = st.columns(2)
            x_range_ht = generate_input_param_range_ui("ht_x", x_param_ht, num_sensitivity_points, ht_rc1)
            y_range_ht = generate_input_param_range_ui("ht_y", y_param_ht, num_sensitivity_points, ht_rc2)
            
            # For heatmaps, Z is implicitly Call Price or Put Price
            call_prices_ht = generate_surface_data(base_params_for_sensitivity_dict, x_param_ht, x_range_ht, y_param_ht, y_range_ht, "Call Price")
            put_prices_ht = generate_surface_data(base_params_for_sensitivity_dict, x_param_ht, x_range_ht, y_param_ht, y_range_ht, "Put Price")

            x_details_ht = get_input_param_details(x_param_ht)
            y_details_ht = get_input_param_details(y_param_ht)
            x_tick_labels = [f"{v:.2f}" if x_param_ht not in [PARAM_VOL, PARAM_RATE] else f"{v*100:.1f}%" for v in x_range_ht]
            y_tick_labels = [f"{v:.2f}" if y_param_ht not in [PARAM_VOL, PARAM_RATE] else f"{v*100:.1f}%" for v in y_range_ht]
            tick_skip = max(1, len(x_range_ht) // 10)

            plot_ht1, plot_ht2 = st.columns(2)
            with plot_ht1:
                st.subheader(f"Call Price vs {x_param_ht.split('(')[0]} & {y_param_ht.split('(')[0]}")
                fig_c_ht, ax_c_ht = plt.subplots(figsize=(8,6)) # Smaller figsize
                sns.heatmap(call_prices_ht, annot=True, fmt=".2f", cmap="Greens", ax=ax_c_ht, annot_kws={"size":6},
                            xticklabels=x_tick_labels[::tick_skip], yticklabels=y_tick_labels[::tick_skip])
                ax_c_ht.set_xlabel(x_details_ht['label']); ax_c_ht.set_ylabel(y_details_ht['label'])
                ax_c_ht.set_xticks(np.arange(len(x_tick_labels))[::tick_skip]+0.5); ax_c_ht.set_yticks(np.arange(len(y_tick_labels))[::tick_skip]+0.5)
                plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); st.pyplot(fig_c_ht)
            with plot_ht2:
                st.subheader(f"Put Price vs {x_param_ht.split('(')[0]} & {y_param_ht.split('(')[0]}")
                fig_p_ht, ax_p_ht = plt.subplots(figsize=(8,6))
                sns.heatmap(put_prices_ht, annot=True, fmt=".2f", cmap="Reds", ax=ax_p_ht, annot_kws={"size":6},
                            xticklabels=x_tick_labels[::tick_skip], yticklabels=y_tick_labels[::tick_skip])
                ax_p_ht.set_xlabel(x_details_ht['label']); ax_p_ht.set_ylabel(y_details_ht['label'])
                ax_p_ht.set_xticks(np.arange(len(x_tick_labels))[::tick_skip]+0.5); ax_p_ht.set_yticks(np.arange(len(y_tick_labels))[::tick_skip]+0.5)
                plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); st.pyplot(fig_p_ht)

    with tab_3d:
        st.header("Surface Plots (3D)")
        st.markdown("Select X-axis, Y-axis (from input parameters) and Z-axis (from output values like Price or Greeks).")
        s3d_c1, s3d_c2, s3d_c3 = st.columns(3)
        x_param_3d = s3d_c1.selectbox("X-axis Input Parameter:", ALL_INPUT_PARAMS, index=0, key="3d_x_param")
        y_param_3d = s3d_c2.selectbox("Y-axis Input Parameter:", ALL_INPUT_PARAMS, index=3, key="3d_y_param")
        z_output_3d_name = s3d_c3.selectbox("Z-axis Output Value:", PLOTTABLE_OUTPUTS_DISPLAY_NAMES, index=0, key="3d_z_param")

        if x_param_3d == y_param_3d: st.error("X and Y axes must be different.")
        else:
            s3d_rc1, s3d_rc2 = st.columns(2)
            x_range_3d = generate_input_param_range_ui("3d_x", x_param_3d, num_sensitivity_points, s3d_rc1)
            y_range_3d = generate_input_param_range_ui("3d_y", y_param_3d, num_sensitivity_points, s3d_rc2)

            z_surface_data = generate_surface_data(base_params_for_sensitivity_dict, x_param_3d, x_range_3d, y_param_3d, y_range_3d, z_output_3d_name)
            
            X_mesh, Y_mesh = np.meshgrid(x_range_3d, y_range_3d)
            x_details_3d = get_input_param_details(x_param_3d)
            y_details_3d = get_input_param_details(y_param_3d)
            z_details_3d = OUTPUT_PARAM_DETAILS[z_output_3d_name]

            # Apply percentage scaling for Volatility/Rate on axes if selected
            x_axis_label_3d = x_details_3d['label']
            if x_param_3d in [PARAM_VOL, PARAM_RATE]: X_mesh_display = X_mesh * 100; x_axis_label_3d += " (%)"
            else: X_mesh_display = X_mesh
            
            y_axis_label_3d = y_details_3d['label']
            if y_param_3d in [PARAM_VOL, PARAM_RATE]: Y_mesh_display = Y_mesh * 100; y_axis_label_3d += " (%)"
            else: Y_mesh_display = Y_mesh
            
            z_axis_label_3d = f"{z_output_3d_name} ({z_details_3d['unit']})" if z_details_3d['unit'] else z_output_3d_name

            st.subheader(f"{z_output_3d_name} Surface")
            fig_3d = go.Figure(data=[go.Surface(z=z_surface_data, x=X_mesh_display, y=Y_mesh_display, colorscale="Viridis", colorbar_title=z_details_3d['unit'] or z_output_3d_name)])
            fig_3d.update_layout(title=f'{z_output_3d_name} vs {x_param_3d.split("(")[0]} & {y_param_3d.split("(")[0]}', 
                                 scene=dict(xaxis_title=x_axis_label_3d, yaxis_title=y_axis_label_3d, zaxis_title=z_axis_label_3d), 
                                 margin=dict(l=0,r=0,b=0,t=40), height=600)
            st.plotly_chart(fig_3d, use_container_width=True)

    with tab_2d_greek:
        st.header("Greek Analysis (2D)")
        st.markdown("Select a Greek and an input parameter to vary. Others fixed from sidebar.")
        g2d_c1, g2d_c2 = st.columns(2)
        selected_greek_name = g2d_c1.selectbox("Select Greek/Output to Plot:", PLOTTABLE_OUTPUTS_DISPLAY_NAMES, index=2, key="2d_greek_name") # Default Call Delta
        varying_input_param_2d = g2d_c2.selectbox("Varying Input Parameter:", ALL_INPUT_PARAMS, index=0, key="2d_varying_param")
        
        varying_range_2d = generate_input_param_range_ui("2d_vary", varying_input_param_2d, num_sensitivity_points, st)
        
        plot_data_1d = generate_1d_plot_data(base_params_for_sensitivity_dict, varying_input_param_2d, varying_range_2d, selected_greek_name)
        
        input_param_details_2d = get_input_param_details(varying_input_param_2d)
        output_details_2d = OUTPUT_PARAM_DETAILS[selected_greek_name]

        x_axis_values_2d = plot_data_1d.index
        x_axis_label_2d = input_param_details_2d['label']
        if varying_input_param_2d in [PARAM_VOL, PARAM_RATE]:
            x_axis_values_2d = x_axis_values_2d * 100
            x_axis_label_2d += " (%)"
        
        y_axis_label_2d = f"{selected_greek_name} ({output_details_2d['unit']})" if output_details_2d['unit'] else selected_greek_name

        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(x=x_axis_values_2d, y=plot_data_1d.values, mode='lines', name=selected_greek_name))
        fig_2d.update_layout(title=f'{selected_greek_name} vs. {varying_input_param_2d.split("(")[0]}', 
                             xaxis_title=x_axis_label_2d, yaxis_title=y_axis_label_2d, height=500,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_2d, use_container_width=True)

else:
    st.warning("Base calculations failed. Please adjust sidebar inputs to enable sensitivity analysis.")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Disclaimer: For educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
