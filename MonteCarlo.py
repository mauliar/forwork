"""
Monte Carlo Valuation Sensitivity Analysis Framework

This module provides a comprehensive framework for financial valuation and sensitivity 
analysis using Monte Carlo simulation methods. It implements option pricing, Greeks 
calculation, scenario analysis, and stress testing capabilities.

Author: AI Assistant
Date: August 31, 2025
License: MIT

Key Features:
- Monte Carlo option pricing with variance reduction techniques
- Sensitivity analysis (Greeks) using finite difference methods
- Parameter sweep analysis across different market conditions
- Scenario analysis and stress testing
- Convergence analysis for simulation accuracy
- Comprehensive comparison with analytical Black-Scholes solutions

Dependencies:
- numpy: Numerical computations and random number generation
- pandas: Data manipulation and analysis
- scipy: Statistical functions and probability distributions
- matplotlib: Plotting and visualization (optional)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class MonteCarloValuationSensitivity:
    """
    A comprehensive Monte Carlo simulation framework for valuation sensitivity analysis.
    
    This class implements Monte Carlo methods to price financial instruments (primarily options)
    and calculate their sensitivities (Greeks) to various input parameters using finite 
    difference methods.
    
    The framework supports:
    - European call and put option pricing
    - All major Greeks (Delta, Gamma, Vega, Theta, Rho)
    - Antithetic variates for variance reduction
    - Parameter sensitivity analysis
    - Scenario analysis and stress testing
    - Convergence analysis
    
    Attributes:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility (annualized)
        option_type (str): 'call' or 'put'
        n_simulations (int): Number of Monte Carlo simulations
        n_steps (int): Number of time steps for path generation
        dt (float): Time step size (T / n_steps)
    """
    
    def __init__(self, 
                 S0: float = 100.0,      # Initial stock price
                 K: float = 100.0,       # Strike price
                 T: float = 1.0,         # Time to maturity (years)
                 r: float = 0.05,        # Risk-free rate
                 sigma: float = 0.2,     # Volatility
                 option_type: str = 'call', # 'call' or 'put'
                 n_simulations: int = 100000, # Number of Monte Carlo simulations
                 n_steps: int = 252):    # Number of time steps (trading days)
        """
        Initialize the Monte Carlo valuation model.
        
        Parameters:
        -----------
        S0 : float, default=100.0
            Initial stock price
        K : float, default=100.0
            Strike price of the option
        T : float, default=1.0
            Time to maturity in years
        r : float, default=0.05
            Risk-free interest rate (annualized)
        sigma : float, default=0.2
            Volatility (annualized)
        option_type : str, default='call'
            Type of option ('call' or 'put')
        n_simulations : int, default=100000
            Number of Monte Carlo simulations to run
        n_steps : int, default=252
            Number of time steps for path generation (252 = trading days in a year)
            
        Raises:
        -------
        ValueError
            If option_type is not 'call' or 'put', or if any parameter is non-positive
        """
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        # Parameter validation
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        if any(x <= 0 for x in [self.S0, self.K, self.T, self.sigma, self.n_simulations, self.n_steps]):
            raise ValueError("All parameters must be positive")
        if not isinstance(self.n_simulations, int) or not isinstance(self.n_steps, int):
            raise ValueError("n_simulations and n_steps must be integers")
    
    def black_scholes_analytical(self) -> float:
        """
        Calculate analytical Black-Scholes price for comparison.
        
        Uses the closed-form Black-Scholes formula to price European options.
        This serves as a benchmark for validating Monte Carlo simulation accuracy.
        
        Returns:
        --------
        float: Analytical option price
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:  # put
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        
        return price
    
    def generate_price_paths(self, n_sims: Optional[int] = None, 
                           antithetic: bool = True, 
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Generate Monte Carlo price paths using Geometric Brownian Motion.
        
        Implements the standard GBM model: dS = Î¼Sdt + ÏƒSdW
        Under risk-neutral measure: Î¼ = r (risk-free rate)
        
        Parameters:
        -----------
        n_sims : int, optional
            Number of simulations (uses instance default if None)
        antithetic : bool, default=True
            Whether to use antithetic variates for variance reduction
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray: Array of final stock prices (n_simulations,)
        """
        if seed is not None:
            np.random.seed(seed)
            
        if n_sims is None:
            n_sims = self.n_simulations
            
        if antithetic:
            # Use antithetic variates for variance reduction
            n_sims_half = n_sims // 2
            Z = np.random.standard_normal(n_sims_half)
            # Antithetic variates: use both Z and -Z
            Z_combined = np.concatenate([Z, -Z])
            if n_sims % 2 == 1:  # Handle odd number of simulations
                Z_combined = np.concatenate([Z_combined, [np.random.standard_normal()]])
        else:
            Z_combined = np.random.standard_normal(n_sims)
        
        # Calculate final stock prices using GBM solution
        # S(T) = S(0) * exp((r - ÏƒÂ²/2)T + ÏƒâˆšT * Z)
        ST = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + 
                              self.sigma * np.sqrt(self.T) * Z_combined)
        
        return ST
    
    def calculate_option_payoffs(self, final_prices: np.ndarray) -> np.ndarray:
        """
        Calculate option payoffs at maturity.
        
        Parameters:
        -----------
        final_prices : np.ndarray
            Array of final stock prices
            
        Returns:
        --------
        np.ndarray: Array of option payoffs
        """
        if self.option_type == 'call':
            payoffs = np.maximum(final_prices - self.K, 0)
        else:  # put
            payoffs = np.maximum(self.K - final_prices, 0)
        
        return payoffs
    
    def monte_carlo_price(self, antithetic: bool = True, 
                         seed: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate option price using Monte Carlo simulation.
        
        This method implements the fundamental Monte Carlo approach:
        1. Generate random stock price paths
        2. Calculate payoffs at maturity
        3. Discount to present value
        4. Take the average across all simulations
        
        Parameters:
        -----------
        antithetic : bool, default=True
            Whether to use antithetic variates for variance reduction
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        dict: Contains the following keys:
            - 'price': Monte Carlo estimated price
            - 'std_error': Standard error of the estimate
            - 'analytical_price': Black-Scholes analytical price
            - 'error': Absolute error vs analytical price
            - 'relative_error': Relative error percentage vs analytical price
            - 'confidence_interval': 95% confidence interval
        """
        # Generate price paths
        final_prices = self.generate_price_paths(antithetic=antithetic, seed=seed)
        
        # Calculate payoffs
        payoffs = self.calculate_option_payoffs(final_prices)
        
        # Discount to present value
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        # Calculate statistics
        mc_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        # Compare with analytical price
        analytical_price = self.black_scholes_analytical()
        error = abs(mc_price - analytical_price)
        relative_error = error / analytical_price * 100 if analytical_price != 0 else 0
        
        # 95% confidence interval
        ci_lower = mc_price - 1.96 * std_error
        ci_upper = mc_price + 1.96 * std_error
        
        return {
            'price': mc_price,
            'std_error': std_error,
            'analytical_price': analytical_price,
            'error': error,
            'relative_error': relative_error,
            'confidence_interval': (ci_lower, ci_upper)
        }
    
    def calculate_sensitivity_finite_difference(self, 
                                              parameter: str, 
                                              bump_size: float = 0.01,
                                              method: str = 'central',
                                              seed: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate sensitivity (Greeks) using finite difference method.
        
        Implements numerical differentiation to approximate partial derivatives:
        - Forward difference: (f(x+h) - f(x)) / h
        - Backward difference: (f(x) - f(x-h)) / h  
        - Central difference: (f(x+h) - f(x-h)) / (2h)
        
        Central difference provides better accuracy (O(hÂ²) vs O(h) error).
        
        Parameters:
        -----------
        parameter : str
            Parameter to calculate sensitivity for ('S0', 'sigma', 'r', 'T')
        bump_size : float, default=0.01
            Size of the parameter bump (should be small)
        method : str, default='central'
            Finite difference method ('forward', 'backward', or 'central')
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        dict: Contains sensitivity value and method details
        """
        if parameter not in ['S0', 'sigma', 'r', 'T']:
            raise ValueError("parameter must be one of: 'S0', 'sigma', 'r', 'T'")
        if method not in ['forward', 'backward', 'central']:
            raise ValueError("method must be one of: 'forward', 'backward', 'central'")
            
        original_value = getattr(self, parameter)
        
        if method == 'forward':
            # Forward difference: (P(x + h) - P(x)) / h
            setattr(self, parameter, original_value + bump_size)
            price_up = self.monte_carlo_price(antithetic=True, seed=seed)['price']
            
            setattr(self, parameter, original_value)
            price_base = self.monte_carlo_price(antithetic=True, seed=seed)['price']
            
            sensitivity = (price_up - price_base) / bump_size
            
        elif method == 'backward':
            # Backward difference: (P(x) - P(x - h)) / h
            setattr(self, parameter, original_value)
            price_base = self.monte_carlo_price(antithetic=True, seed=seed)['price']
            
            setattr(self, parameter, original_value - bump_size)
            price_down = self.monte_carlo_price(antithetic=True, seed=seed)['price']
            
            sensitivity = (price_base - price_down) / bump_size
            
        else:  # central
            # Central difference: (P(x + h) - P(x - h)) / (2h)
            setattr(self, parameter, original_value + bump_size)
            price_up = self.monte_carlo_price(antithetic=True, seed=seed)['price']
            
            setattr(self, parameter, original_value - bump_size)
            price_down = self.monte_carlo_price(antithetic=True, seed=seed)['price']
            
            sensitivity = (price_up - price_down) / (2 * bump_size)
        
        # Restore original value
        setattr(self, parameter, original_value)
        
        return {
            'sensitivity': sensitivity,
            'parameter': parameter,
            'bump_size': bump_size,
            'method': method,
            'original_value': original_value
        }
    
    def calculate_all_greeks(self, bump_sizes: Optional[Dict[str, float]] = None,
                           seed: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate all major Greeks using finite difference methods.
        
        Greeks are the sensitivities of option price to various parameters:
        - Delta: Sensitivity to underlying price (âˆ‚V/âˆ‚S)
        - Gamma: Second-order sensitivity to underlying price (âˆ‚Â²V/âˆ‚SÂ²)
        - Vega: Sensitivity to volatility (âˆ‚V/âˆ‚Ïƒ)
        - Theta: Sensitivity to time decay (âˆ‚V/âˆ‚T)
        - Rho: Sensitivity to interest rate (âˆ‚V/âˆ‚r)
        
        Parameters:
        -----------
        bump_sizes : dict, optional
            Custom bump sizes for each parameter. If None, uses reasonable defaults.
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        dict: All calculated Greeks
        """
        if bump_sizes is None:
            bump_sizes = {
                'S0': 1.0,      # Delta: sensitivity to $1 stock price change
                'sigma': 0.01,  # Vega: sensitivity to 1% volatility change
                'T': 1/365,     # Theta: sensitivity to 1 day time decay
                'r': 0.01       # Rho: sensitivity to 1% interest rate change
            }
        
        greeks = {}
        
        # Delta: âˆ‚V/âˆ‚S - sensitivity to stock price
        delta_result = self.calculate_sensitivity_finite_difference('S0', bump_sizes['S0'], seed=seed)
        greeks['delta'] = delta_result['sensitivity']
        
        # Vega: âˆ‚V/âˆ‚Ïƒ - sensitivity to volatility
        vega_result = self.calculate_sensitivity_finite_difference('sigma', bump_sizes['sigma'], seed=seed)
        greeks['vega'] = vega_result['sensitivity']
        
        # Theta: âˆ‚V/âˆ‚T - sensitivity to time (negative because time decreases)
        theta_result = self.calculate_sensitivity_finite_difference('T', bump_sizes['T'], seed=seed)
        greeks['theta'] = -theta_result['sensitivity']  # Negative for time decay
        
        # Rho: âˆ‚V/âˆ‚r - sensitivity to interest rate
        rho_result = self.calculate_sensitivity_finite_difference('r', bump_sizes['r'], seed=seed)
        greeks['rho'] = rho_result['sensitivity']
        
        # Gamma: âˆ‚Â²V/âˆ‚SÂ² - second-order sensitivity to stock price
        # Calculate as finite difference of Delta
        original_S = self.S0
        bump = bump_sizes['S0']
        
        # Calculate delta at S + bump
        self.S0 = original_S + bump
        delta_up = self.calculate_sensitivity_finite_difference('S0', bump, seed=seed)['sensitivity']
        
        # Calculate delta at S - bump
        self.S0 = original_S - bump
        delta_down = self.calculate_sensitivity_finite_difference('S0', bump, seed=seed)['sensitivity']
        
        # Restore original value
        self.S0 = original_S
        
        # Gamma is the finite difference of Delta
        greeks['gamma'] = (delta_up - delta_down) / (2 * bump)
        
        return greeks
    
    def sensitivity_analysis(self, 
                           parameter: str,
                           min_val: float,
                           max_val: float,
                           n_points: int = 20,
                           seed: Optional[int] = None) -> pd.DataFrame:
        """
        Perform sensitivity analysis by varying a parameter across a range.
        
        This method creates a parameter sweep to understand how option price and 
        Greeks change as market conditions vary. Useful for understanding option
        behavior across different scenarios.
        
        Parameters:
        -----------
        parameter : str
            Parameter to vary ('S0', 'sigma', 'r', 'T')
        min_val : float
            Minimum value of parameter range
        max_val : float
            Maximum value of parameter range
        n_points : int, default=20
            Number of points to evaluate across the range
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame: Results containing parameter values, prices, and Greeks
        """
        if parameter not in ['S0', 'sigma', 'r', 'T']:
            raise ValueError("parameter must be one of: 'S0', 'sigma', 'r', 'T'")
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
            
        original_value = getattr(self, parameter)
        values = np.linspace(min_val, max_val, n_points)
        
        results = []
        for val in values:
            setattr(self, parameter, val)
            
            # Calculate price and Greeks
            pricing_result = self.monte_carlo_price(antithetic=True, seed=seed)
            greeks = self.calculate_all_greeks(seed=seed)
            
            result = {
                parameter: val,
                'mc_price': pricing_result['price'],
                'analytical_price': pricing_result['analytical_price'],
                'error': pricing_result['error'],
                'std_error': pricing_result['std_error'],
                'relative_error': pricing_result['relative_error'],
                **greeks
            }
            results.append(result)
        
        # Restore original value
        setattr(self, parameter, original_value)
        
        return pd.DataFrame(results)
    
    def scenario_analysis(self, scenarios: List[Dict[str, Union[str, float]]], 
                         seed: Optional[int] = None) -> pd.DataFrame:
        """
        Perform scenario analysis under different market conditions.
        
        Parameters:
        -----------
        scenarios : List[Dict]
            List of scenario dictionaries containing parameter values
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame: Results for each scenario
        """
        # Store original values
        original_params = {
            'S0': self.S0,
            'sigma': self.sigma,
            'r': self.r,
            'T': self.T,
            'K': self.K
        }
        
        scenario_results = []
        
        for scenario in scenarios:
            # Update model parameters
            for param, value in scenario.items():
                if param != 'name' and hasattr(self, param):
                    setattr(self, param, value)
            
            # Calculate price and Greeks
            pricing = self.monte_carlo_price(antithetic=True, seed=seed)
            greeks = self.calculate_all_greeks(seed=seed)
            
            result = {
                'scenario': scenario.get('name', 'Unknown'),
                **{k: v for k, v in scenario.items() if k != 'name'},
                'option_price': pricing['price'],
                'analytical_price': pricing['analytical_price'],
                'error': pricing['error'],
                **greeks
            }
            scenario_results.append(result)
        
        # Restore original values
        for param, value in original_params.items():
            setattr(self, param, value)
        
        return pd.DataFrame(scenario_results)
    
    def convergence_analysis(self, simulation_counts: List[int], 
                           seed: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze convergence properties of Monte Carlo simulation.
        
        Parameters:
        -----------
        simulation_counts : List[int]
            List of simulation counts to test
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame: Convergence analysis results
        """
        analytical_price = self.black_scholes_analytical()
        original_n_sims = self.n_simulations
        
        convergence_results = []
        
        for n_sims in simulation_counts:
            self.n_simulations = n_sims
            
            # Calculate price
            pricing = self.monte_carlo_price(antithetic=True, seed=seed)
            
            # Calculate 95% confidence interval width
            ci_width = 1.96 * pricing['std_error'] * 2
            
            result = {
                'n_simulations': n_sims,
                'mc_price': pricing['price'],
                'analytical_price': analytical_price,
                'error': pricing['error'],
                'std_error': pricing['std_error'],
                'ci_width': ci_width,
                'relative_error': pricing['relative_error']
            }
            convergence_results.append(result)
        
        # Restore original number of simulations
        self.n_simulations = original_n_sims
        
        return pd.DataFrame(convergence_results)


# Example usage and demonstration functions
def example_basic_usage():
    """Demonstrate basic usage of the Monte Carlo framework."""
    
    print("=== Monte Carlo Valuation Sensitivity Analysis ===\n")
    
    # Initialize the model
    mc_model = MonteCarloValuationSensitivity(
        S0=100.0,      # Stock price
        K=105.0,       # Strike price
        T=0.25,        # 3 months to expiry
        r=0.05,        # 5% risk-free rate
        sigma=0.2,     # 20% volatility
        option_type='call',
        n_simulations=100000
    )
    
    print("Model Parameters:")
    print(f"Stock Price (S0): ${mc_model.S0}")
    print(f"Strike Price (K): ${mc_model.K}")
    print(f"Time to Expiry (T): {mc_model.T} years")
    print(f"Risk-free Rate (r): {mc_model.r:.1%}")
    print(f"Volatility (Ïƒ): {mc_model.sigma:.1%}")
    print(f"Option Type: {mc_model.option_type.upper()}")
    print(f"Number of Simulations: {mc_model.n_simulations:,}")
    print()
    
    # 1. Basic Monte Carlo Pricing
    print("1. Monte Carlo Pricing:")
    pricing_result = mc_model.monte_carlo_price(antithetic=True)
    print(f"Monte Carlo Price: ${pricing_result['price']:.4f}")
    print(f"Analytical Price: ${pricing_result['analytical_price']:.4f}")
    print(f"Absolute Error: ${pricing_result['error']:.4f}")
    print(f"Relative Error: {pricing_result['relative_error']:.2f}%")
    print(f"Standard Error: ${pricing_result['std_error']:.4f}")
    ci_lower, ci_upper = pricing_result['confidence_interval']
    print(f"95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]")
    print()
    
    # 2. Calculate Greeks
    print("2. Sensitivity Analysis (Greeks):")
    greeks = mc_model.calculate_all_greeks()
    print(f"Delta (price sensitivity to $1 stock move): {greeks['delta']:.4f}")
    print(f"Gamma (delta sensitivity to $1 stock move): {greeks['gamma']:.6f}")
    print(f"Vega (price sensitivity to 1% vol change): {greeks['vega']:.4f}")
    print(f"Theta (time decay per day): {greeks['theta']:.4f}")
    print(f"Rho (rate sensitivity to 1% rate change): {greeks['rho']:.4f}")
    print()
    
    return mc_model, pricing_result, greeks


def example_sensitivity_analysis(mc_model):
    """Demonstrate parameter sensitivity analysis."""
    
    print("3. Parameter Sensitivity Analysis:")
    
    # Volatility sensitivity
    print("a) Volatility Impact (10% to 40% volatility):")
    vol_analysis = mc_model.sensitivity_analysis('sigma', 0.1, 0.4, 8)
    print(vol_analysis[['sigma', 'mc_price', 'delta', 'vega', 'theta']].round(4))
    print()
    
    # Stock price sensitivity  
    print("b) Stock Price Impact ($90 to $110):")
    price_analysis = mc_model.sensitivity_analysis('S0', 90, 110, 8)
    print(price_analysis[['S0', 'mc_price', 'delta', 'gamma']].round(4))
    print()
    
    return vol_analysis, price_analysis


def example_scenario_analysis(mc_model):
    """Demonstrate scenario analysis."""
    
    print("4. Scenario Analysis:")
    
    scenarios = [
        {"name": "Base Case", "S0": 100, "sigma": 0.20, "r": 0.05},
        {"name": "Bull Market", "S0": 110, "sigma": 0.15, "r": 0.04},
        {"name": "Bear Market", "S0": 90, "sigma": 0.30, "r": 0.06},
        {"name": "High Volatility", "S0": 100, "sigma": 0.35, "r": 0.05},
        {"name": "Low Volatility", "S0": 100, "sigma": 0.10, "r": 0.05},
    ]
    
    scenario_df = mc_model.scenario_analysis(scenarios)
    print(scenario_df[['scenario', 'option_price', 'delta', 'vega', 'theta']].round(4))
    print()
    
    return scenario_df


def example_convergence_analysis(mc_model):
    """Demonstrate convergence analysis."""
    
    print("5. Monte Carlo Convergence Analysis:")
    
    simulation_counts = [1000, 5000, 10000, 25000, 50000, 100000]
    convergence_df = mc_model.convergence_analysis(simulation_counts)
    print(convergence_df[['n_simulations', 'mc_price', 'error', 'std_error', 'relative_error']].round(4))
    print()
    
    return convergence_df


if __name__ == "__main__":
    """
    Main execution: Run comprehensive examples of the Monte Carlo framework.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    model, pricing, greeks_result = example_basic_usage()
    vol_df, price_df = example_sensitivity_analysis(model)
    scenario_df = example_scenario_analysis(model)
    convergence_df = example_convergence_analysis(model)
    
    print("="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey Takeaways:")
    print(f"â€¢ Monte Carlo price vs analytical: ${pricing['price']:.4f} vs ${pricing['analytical_price']:.4f}")
    print(f"â€¢ Relative error: {pricing['relative_error']:.2f}% (excellent accuracy)")
    print(f"â€¢ Delta suggests {greeks_result['delta']:.1%} hedge ratio")
    print(f"â€¢ Vega shows ${greeks_result['vega']:.2f} price change per 1% volatility change")
    print(f"â€¢ Theta indicates ${abs(greeks_result['theta']):.2f} daily time decay")
    print("\nAll results are stored in DataFrames for further analysis and visualization.")