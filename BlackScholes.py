from numpy import exp, sqrt, log
from scipy.stats import norm


class BlackScholes:
    """
    Calculates European call/put option prices and all standard Greeks using the Black-Scholes model.

    The model assumes no dividends are paid by the underlying asset.
    Calculated values are stored as instance attributes.
    """
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        """
        Initializes the Black-Scholes model with the required parameters.

        Args:
            time_to_maturity (float): Time to maturity of the option in years (e.g., 0.5 for 6 months).
                                      Must be greater than 0.
            strike (float): Strike price of the option. Must be positive.
            current_price (float): Current price of the underlying asset. Must be positive.
            volatility (float): Annualized volatility of the underlying asset's returns
                                (e.g., 0.2 for 20%). Must be greater than 0.
            interest_rate (float): Annualized risk-free interest rate (e.g., 0.05 for 5%).

        Instance Attributes (populated after run_calculations()):
            d1 (float): Intermediate parameter d1.
            d2 (float): Intermediate parameter d2.
            call_price (float): Calculated European call option price.
            put_price (float): Calculated European put option price.
            call_delta (float): Delta of the call option.
            put_delta (float): Delta of the put option.
            gamma (float): Gamma of both call and put options.
            vega (float): Vega of both call and put options (sensitivity to 1 unit change in volatility).
            call_theta (float): Theta of the call option (annualized).
            put_theta (float): Theta of the put option (annualized).
            call_rho (float): Rho of the call option (sensitivity to 1 unit change in interest rate).
            put_rho (float): Rho of the put option (sensitivity to 1 unit change in interest rate).
        """
        if time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive.")
        if strike <= 0:
            raise ValueError("Strike price must be positive.")
        if current_price <= 0:
            raise ValueError("Current asset price must be positive.")
        if volatility <= 0:
            raise ValueError("Volatility must be positive.")
        # Interest rate can theoretically be negative, zero or positive.

        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

        # Attributes to store calculated values
        self.d1: float = 0.0
        self.d2: float = 0.0
        self.call_price: float = 0.0
        self.put_price: float = 0.0
        self.call_delta: float = 0.0
        self.put_delta: float = 0.0
        self.gamma: float = 0.0  # Gamma is the same for calls and puts
        self.vega: float = 0.0   # Vega is the same for calls and puts
        self.call_theta: float = 0.0
        self.put_theta: float = 0.0
        self.call_rho: float = 0.0
        self.put_rho: float = 0.0

    def _calculate_d1_d2(self):
        """
        Calculates the intermediate d1 and d2 parameters for the Black-Scholes formula.
        These are stored as instance attributes self.d1 and self.d2.
        """
        # __init__ ensures time_to_maturity > 0 and volatility > 0.
        # Therefore, self.volatility * sqrt(self.time_to_maturity) will not be zero.
        sqrt_T = sqrt(self.time_to_maturity)
        denominator = self.volatility * sqrt_T
        
        # Handle case where S or K might be zero if not caught by __init__ (though current __init__ prevents this)
        # log(self.current_price / self.strike) can cause issues if current_price or strike is zero or negative.
        # Current __init__ checks S > 0 and K > 0.

        self.d1 = (
            log(self.current_price / self.strike) +
            (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
        ) / denominator

        self.d2 = self.d1 - denominator # d2 = d1 - sigma * sqrt(T)

    def run_calculations(self):
        """
        Performs all Black-Scholes calculations (prices and Greeks)
        and stores them as instance attributes.
        """
        self._calculate_d1_d2()

        S = self.current_price
        K = self.strike
        T = self.time_to_maturity
        sigma = self.volatility
        r = self.interest_rate

        sqrt_T = sqrt(T)
        exp_rt = exp(-r * T) # Discount factor e^(-rT)
        K_exp_rt = K * exp_rt # Present value of strike price

        # Cumulative distribution functions (CDF) for N(d1), N(d2), N(-d1), N(-d2)
        N_d1 = norm.cdf(self.d1)
        N_d2 = norm.cdf(self.d2)
        N_neg_d1 = norm.cdf(-self.d1)
        N_neg_d2 = norm.cdf(-self.d2)

        # Probability density function (PDF) for N'(d1)
        N_prime_d1 = norm.pdf(self.d1)

        # --- Option Prices ---
        self.call_price = S * N_d1 - K_exp_rt * N_d2
        self.put_price = K_exp_rt * N_neg_d2 - S * N_neg_d1

        # --- GREEKS ---

        # Delta: Rate of change of option price with respect to asset price
        # Call Delta: N(d1)
        # Put Delta: N(d1) - 1  (or -N(-d1))
        self.call_delta = N_d1
        self.put_delta = N_d1 - 1

        # Gamma: Rate of change of delta with respect to asset price
        # Same for call and put options
        # Formula: N'(d1) / (S * sigma * sqrt(T))
        # Denominator S * sigma * sqrt_T is guaranteed non-zero by __init__ checks.
        self.gamma = N_prime_d1 / (S * sigma * sqrt_T)

        # Vega: Rate of change of option price with respect to volatility
        # Same for call and put options
        # Formula: S * N'(d1) * sqrt(T)
        # This is the change in option price for a 1 unit (100%) change in volatility.
        self.vega = S * N_prime_d1 * sqrt_T

        # Theta: Rate of change of option price with respect to time (time decay)
        # Formulas give annualized theta. For daily theta, divide by 365 or 252.
        # Call Theta: -(S * N'(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-rT) * N(d2)
        # Put Theta:  -(S * N'(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-rT) * N(-d2)
        common_theta_term = -(S * N_prime_d1 * sigma) / (2 * sqrt_T)
        self.call_theta = common_theta_term - r * K_exp_rt * N_d2
        self.put_theta = common_theta_term + r * K_exp_rt * N_neg_d2

        # Rho: Rate of change of option price with respect to interest rate
        # Formulas give rho for a 1 unit (100%) change in interest rate.
        # Call Rho: K * T * exp(-rT) * N(d2)
        # Put Rho: -K * T * exp(-rT) * N(-d2)
        self.call_rho = T * K_exp_rt * N_d2
        self.put_rho = -T * K_exp_rt * N_neg_d2


if __name__ == "__main__":
    # Example parameters (prices assumed to be in INR)
    time_to_maturity_years = 2.0  # 2 years
    strike_price_inr = 9000.0     # Strike price in INR
    current_asset_price_inr = 10000.0 # Current asset price in INR
    asset_volatility = 0.20       # 20% annualized volatility
    risk_free_rate = 0.05         # 5% annualized risk-free rate

    print(f"Calculating Black-Scholes for:")
    print(f"  Time to Maturity: {time_to_maturity_years} years")
    print(f"  Strike Price: {strike_price_inr:.2f} INR")
    print(f"  Current Asset Price: {current_asset_price_inr:.2f} INR")
    print(f"  Volatility: {asset_volatility*100:.2f}%")
    print(f"  Risk-Free Interest Rate: {risk_free_rate*100:.2f}%\n")

    # Initialize Black-Scholes model
    try:
        bs_model = BlackScholes(
            time_to_maturity=time_to_maturity_years,
            strike=strike_price_inr,
            current_price=current_asset_price_inr,
            volatility=asset_volatility,
            interest_rate=risk_free_rate
        )

        # Run the calculations
        bs_model.run_calculations()

        # Print the results
        print("--- Option Prices ---")
        print(f"Call Option Price: {bs_model.call_price:.2f} INR")
        print(f"Put Option Price:  {bs_model.put_price:.2f} INR")
        print("\n--- Option Greeks (Standard Interpretation) ---")
        print(f"Call Delta: {bs_model.call_delta:.4f}")
        print(f"Put Delta:  {bs_model.put_delta:.4f}")
        print(f"Gamma (for both call/put): {bs_model.gamma:.6f}") # Gamma is often small
        
        # Vega: change per 1% point change in volatility (volatility is 0.20, so 1% point is 0.01)
        print(f"Vega (per 1% vol change, for both call/put): {bs_model.vega * 0.01:.4f} INR")
        
        # Theta: change per calendar day (annualized Theta / 365)
        print(f"Call Theta (per day): {bs_model.call_theta / 365:.4f} INR")
        print(f"Put Theta (per day):  {bs_model.put_theta / 365:.4f} INR")
        
        # Rho: change per 1% point change in interest rate (interest rate is 0.05, so 1% point is 0.01)
        print(f"Call Rho (per 1% rate change): {bs_model.call_rho * 0.01:.4f} INR")
        print(f"Put Rho (per 1% rate change):  {bs_model.put_rho * 0.01:.4f} INR")

        print("\n--- Option Greeks (Raw Derivatives) ---")
        print(f"Raw Vega (derivative w.r.t. volatility): {bs_model.vega:.4f}")
        print(f"Raw Call Theta (annualized): {bs_model.call_theta:.4f}")
        print(f"Raw Put Theta (annualized): {bs_model.put_theta:.4f}")
        print(f"Raw Call Rho (derivative w.r.t. interest rate): {bs_model.call_rho:.4f}")
        print(f"Raw Put Rho (derivative w.r.t. interest rate): {bs_model.put_rho:.4f}")


    except ValueError as e:
        print(f"Error initializing BlackScholes model: {e}")

    # Example with invalid input to demonstrate validation
    print("\n--- Example with invalid input (Time to maturity = 0) ---")
    try:
        invalid_bs_model = BlackScholes(
            time_to_maturity=0,
            strike=100,
            current_price=100,
            volatility=0.2,
            interest_rate=0.05
        )
        invalid_bs_model.run_calculations() # This line won't be reached
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Example with invalid input (Volatility = 0) ---")
    try:
        invalid_bs_model_vol = BlackScholes(
            time_to_maturity=1,
            strike=100,
            current_price=100,
            volatility=0,
            interest_rate=0.05
        )
        invalid_bs_model_vol.run_calculations() # This line won't be reached
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Example with invalid input (Current Price = 0) ---")
    try:
        invalid_bs_model_price = BlackScholes(
            time_to_maturity=1,
            strike=100,
            current_price=0,
            volatility=0.2,
            interest_rate=0.05
        )
        invalid_bs_model_price.run_calculations() # This line won't be reached
    except ValueError as e:
        print(f"Caught expected error: {e}")
