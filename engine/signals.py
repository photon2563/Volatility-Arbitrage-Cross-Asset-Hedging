"""
Signal Generation Engine based on Simon & Campasano Basis Predictability.

The engine calculates the daily normalized roll:
    Roll(t) = (VIX_F(t) - VIX_S(t)) / TTS(t)
which measures the daily premium or discount amortized over the remaining life of the contract.

Trading Rules:
- Contango Strategy (Short): When Roll(t) > tau_upper, initiate short VIX futures position (-1).
- Backwardation Strategy (Long): When Roll(t) < tau_lower, initiate long VIX futures position (+1).
- Neutral Band Exit Rule: When Roll(t) falls back within [tau_lower, tau_upper], immediately close position (0).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TradeSignal:
    date: str
    roll: float
    signal: int          # -1 (Short VX), +1 (Long VX), 0 (Neutral / Close)
    transition: str      # E.g., 'ENTRY_SHORT', 'EXIT_NEUTRAL', 'HOLD_SHORT', etc.
    justification: str   # Human-readable justification for the explainability dashboard


class SignalEngine:
    """
    Evaluates the VIX futures term structure and generates directional trading signals.
    """
    def __init__(
        self,
        tau_upper: float = 0.08,   # Upper threshold for contango shorting
        tau_lower: float = -0.05,  # Lower threshold for backwardation buying
    ):
        self.tau_upper = tau_upper
        self.tau_lower = tau_lower

    def generate_signals(
        self,
        dates: Any,
        unadj_vx_price: np.ndarray,
        spot_vix: np.ndarray,
        unadj_tts: np.ndarray
    ) -> List[TradeSignal]:
        """
        Generates daily trading signals and explainability justifications across the time series.
        """
        n = len(unadj_vx_price)
        signals = []
        current_pos = 0
        
        for t in range(n):
            date_str = str(dates[t])[:10]
            vx = unadj_vx_price[t]
            v_spot = spot_vix[t]
            tts = max(int(unadj_tts[t]), 1)  # prevent division by zero
            
            # Normalized daily roll
            roll = (vx - v_spot) / float(tts)
            
            # State machine & transition determination
            new_pos = current_pos
            transition = "HOLD_NEUTRAL"
            justification = ""
            
            if current_pos == 0:
                if roll > self.tau_upper:
                    new_pos = -1
                    transition = "ENTRY_SHORT"
                    justification = (
                        f"The VIX term structure is exhibiting steep contango. Normalized daily roll is {roll:.4f}, "
                        f"which breaches the upper threshold of {self.tau_upper:.4f}. This indicates statistical "
                        f"overpricing of the VX future relative to spot VIX. Historical probability favors downward "
                        f"mean reversion. Directional action authorized: Initiate Short Position in VX Future."
                    )
                elif roll < self.tau_lower:
                    new_pos = 1
                    transition = "ENTRY_LONG"
                    justification = (
                        f"The VIX term structure has inverted into backwardation. Normalized daily roll is {roll:.4f}, "
                        f"falling below the lower threshold of {self.tau_lower:.4f}. VX future trades at a discount "
                        f"to spot VIX during acute market panic. Directional action authorized: Initiate Long Position in VX Future."
                    )
                else:
                    transition = "HOLD_NEUTRAL"
                    justification = (
                        f"Normalized daily roll ({roll:.4f}) is within the neutral band "
                        f"[{self.tau_lower:.4f}, {self.tau_upper:.4f}]. No structural VRP edge present. Standby."
                    )
            elif current_pos == -1:  # Currently short contango
                if roll <= self.tau_upper and roll >= self.tau_lower:
                    new_pos = 0
                    transition = "EXIT_NEUTRAL"
                    justification = (
                        f"Initial condition exit rule triggered. Favorable contango roll ({roll:.4f}) has decayed "
                        f"back into the neutral band [{self.tau_lower:.4f}, {self.tau_upper:.4f}]. Exiting short "
                        f"position to lock in harvested roll yield and eliminate unnecessary tail exposure."
                    )
                elif roll < self.tau_lower:
                    new_pos = 1
                    transition = "REVERSAL_LONG"
                    justification = (
                        f"Market regime shift! Contango inverted directly into backwardation (Roll = {roll:.4f} < {self.tau_lower:.4f}). "
                        f"Closing short position and immediately reversing to Long VX Future."
                    )
                else:
                    transition = "HOLD_SHORT"
                    justification = (
                        f"Contango remains strong (Roll = {roll:.4f} > {self.tau_upper:.4f}). Continuing to harvest "
                        f"negative roll yield from long-only VIX option buyers. Holding Short VX Future."
                    )
            elif current_pos == 1:  # Currently long backwardation
                if roll >= self.tau_lower and roll <= self.tau_upper:
                    new_pos = 0
                    transition = "EXIT_NEUTRAL"
                    justification = (
                        f"Initial condition exit rule triggered. Backwardation discount ({roll:.4f}) has dissipated "
                        f"back into neutral band [{self.tau_lower:.4f}, {self.tau_upper:.4f}] as panic subsides. "
                        f"Closing long position to capture mean-reversion gain."
                    )
                elif roll > self.tau_upper:
                    new_pos = -1
                    transition = "REVERSAL_SHORT"
                    justification = (
                        f"Market regime shift! Curve normalized into steep contango (Roll = {roll:.4f} > {self.tau_upper:.4f}). "
                        f"Closing long position and reversing to Short VX Future to harvest VRP."
                    )
                else:
                    transition = "HOLD_LONG"
                    justification = (
                        f"Curve remains inverted (Roll = {roll:.4f} < {self.tau_lower:.4f}). Holding Long VX Future "
                        f"for continued distress protection / mean reversion."
                    )
                    
            signals.append(TradeSignal(
                date=date_str,
                roll=roll,
                signal=new_pos,
                transition=transition,
                justification=justification
            ))
            current_pos = new_pos
            
        return signals
