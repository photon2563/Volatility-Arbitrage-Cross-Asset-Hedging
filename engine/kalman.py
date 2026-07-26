"""
Dynamic Hedging Engine using a Linear Gaussian Kalman Filter State-Space Model.

This engine addresses time-varying cointegration and non-stationarity between VIX futures (VX)
and S&P 500 E-mini futures (ES) by treating their relationship as an unobservable, evolving state:
    Observation Eq: y(t) = beta(t) * x(t) + e(t),    e(t) ~ N(0, V_e)
    Transition Eq:  beta(t) = beta(t-1) + w(t),      w(t) ~ N(0, V_w)

Where:
    y(t) = VIX futures daily return
    x(t) = S&P 500 E-mini daily return
    beta(t) = Dynamic hedge ratio (statistically negative, e.g., -0.75)
    e(t) = Measurement noise (options market microstructure frictions / independent order flow)
    w(t) = Process noise (speed of correlation drift / structural regime shifts)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class KalmanState:
    date: str
    beta_prior: float      # beta_{t|t-1}
    p_prior: float         # P_{t|t-1} (prior state variance)
    innovation: float      # e_t = y_t - beta_{t|t-1} * x_t
    innovation_var: float  # S_t = x_t^2 * P_{t|t-1} + V_e
    kalman_gain: float     # K_t = P_{t|t-1} * x_t / S_t
    beta_post: float       # beta_{t|t} (optimal posteriori dynamic hedge ratio)
    p_post: float          # P_{t|t} (posteriori state variance / uncertainty metric)
    z_score: float         # innovation / sqrt(innovation_var), used for risk scaling!
    es_contracts_req: float # Required offsetting ES contract volume per VX contract


class KalmanHedgeEngine:
    """
    Recursive Kalman filter for dynamic beta estimation and position sizing.
    """
    def __init__(
        self,
        v_e: float = 1e-3,       # Measurement noise variance (microstructure noise)
        v_w: float = 1e-5,       # Process noise variance (random walk state drift speed)
        init_beta: float = -0.75, # Initial beta prior (VX and ES are negatively correlated)
        init_p: float = 1.0,     # Initial state uncertainty
        vx_multiplier: float = 1000.0, # VX dollar value per index point ($1,000)
        es_multiplier: float = 50.0    # ES dollar value per index point ($50)
    ):
        self.v_e = v_e
        self.v_w = v_w
        self.beta_post = init_beta
        self.p_post = init_p
        self.vx_multiplier = vx_multiplier
        self.es_multiplier = es_multiplier

    def filter_series(
        self,
        dates: Any,
        vx_returns: np.ndarray,
        es_returns: np.ndarray,
        es_prices: np.ndarray,
        vx_prices: np.ndarray,
        vx_positions: np.ndarray
    ) -> List[KalmanState]:
        """
        Runs the recursive prediction-update cycle across the entire time series.
        Calculates the required offsetting ES contract volume to achieve delta neutrality.
        """
        n = len(vx_returns)
        history = []
        
        for t in range(n):
            date_str = str(dates[t])[:10]
            y_t = vx_returns[t]
            x_t = es_returns[t]
            es_p = es_prices[t]
            vx_p = vx_prices[t]
            vx_pos = vx_positions[t]
            
            # --- PREDICTION PHASE ---
            # Random walk assumption: a priori state equals previous a posteriori state
            beta_prior = self.beta_post
            # Prior variance increases due to process noise
            p_prior = self.p_post + self.v_w
            
            # --- UPDATE PHASE ---
            # Measurement prediction & innovation (error)
            y_pred = beta_prior * x_t
            e_t = y_t - y_pred
            
            # Innovation variance S_t = x_t^2 * P_{t|t-1} + V_e
            s_t = (x_t ** 2) * p_prior + self.v_e
            
            # Kalman Gain K_t = P_{t|t-1} * x_t / S_t
            # Acts as an algorithmic learning rate: high S_t -> small K_t (ignore outlier); high P_prior -> large K_t (update aggressively)
            k_t = (p_prior * x_t) / s_t if s_t != 0 else 0.0
            
            # Posterior state update
            beta_post = beta_prior + k_t * e_t
            # Posterior variance update (uncertainty reduction)
            p_post = (1.0 - k_t * x_t) * p_prior
            
            # Z-score of innovation (standard deviations from expectation)
            std_innov = np.sqrt(s_t)
            z_score = e_t / std_innov if std_innov != 0 else 0.0
            
            # --- CONTRACT MULTIPLIER & DELTA NEUTRALITY SIZING ---
            # To neutralize 1 unit of VX return variance, we need (-beta) units of ES return variance.
            # Converted to physical contract volumes per 1 VX contract:
            # Notional value of 1 VX contract = vx_price * $1,000.
            # Notional value of 1 ES contract = es_price * $50.
            # Delta-neutral ES contracts per 1 VX contract = - vx_pos * beta_post * (VX_Notional / ES_Notional)
            vx_notional = vx_p * self.vx_multiplier
            es_notional = es_p * self.es_multiplier
            es_contracts = - vx_pos * beta_post * (vx_notional / es_notional) if es_notional != 0 else 0.0
            
            # Save state
            self.beta_post = beta_post
            self.p_post = p_post
            
            history.append(KalmanState(
                date=date_str,
                beta_prior=beta_prior,
                p_prior=p_prior,
                innovation=e_t,
                innovation_var=s_t,
                kalman_gain=k_t,
                beta_post=beta_post,
                p_post=p_post,
                z_score=z_score,
                es_contracts_req=es_contracts
            ))
            
        return history
