"""Models.

Changes affecting results or their presentation should also update
parameters.py `change_date`, so users can see when results have last
changed
"""

from __future__ import annotations

from typing import Dict, Generator, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .parameters import Parameters


class SimSirModel:

    def __init__(self, p: Parameters) -> SimSirModel:
        # TODO missing initial recovered value
        susceptible = p.susceptible
        recovered = 0.0
        recovery_days = p.recovery_days

        rates = {
            key: d.rate
            for key, d in p.dispositions.items()
        }

        lengths_of_stay = {
            key: d.length_of_stay
            for key, d in p.dispositions.items()
        }

        # Note: this should not be an integer.
        # We're appoximating infected from what we do know.
        # TODO market_share > 0, hosp_rate > 0
        infected = (
            p.current_hospitalized / p.market_share / p.hospitalized.rate
        )

        detection_probability = (
            p.known_infected / infected if infected > 1.0e-7 else None
        )

        intrinsic_growth_rate = \
            (2.0 ** (1.0 / p.doubling_time) - 1.0) if p.doubling_time > 0.0 else 0.0

        gamma = 1.0 / recovery_days

        self.omega = omega = 1.0 - p.old_pop_relative_contact_rate

        # Contact rate, beta
        beta = (
            (intrinsic_growth_rate + gamma)
            / susceptible
            * (1.0 - p.relative_contact_rate)
        )  # {rate based on doubling time} / {initial susceptible}

        # r_t is r_0 after distancing
        r_t = beta / gamma * susceptible

        # Simplify equation to avoid division by zero:
        # self.r_naught = r_t / (1.0 - relative_contact_rate)
        r_naught = (intrinsic_growth_rate + gamma) / gamma
        doubling_time_t = 1.0 / np.log2(
            beta * susceptible - gamma + 1)

        raw_df = sim_sir_df(
            susceptible,
            infected,
            recovered,
            p.older_population_rate,
            beta,
            omega,
            gamma,
            p.n_days,
        )
        dispositions_df = build_dispositions_df(raw_df, rates, p.market_share)
        admits_df = build_admits_df(dispositions_df)
        census_df = build_census_df(admits_df, lengths_of_stay)

        self.susceptible = susceptible
        self.infected = infected
        self.recovered = recovered

        self.detection_probability = detection_probability
        self.recovered = recovered
        self.intrinsic_growth_rate = intrinsic_growth_rate
        self.gamma = gamma
        self.beta = beta
        self.r_t = r_t
        self.r_naught = r_naught
        self.doubling_time_t = doubling_time_t
        self.raw_df = raw_df
        self.dispositions_df = dispositions_df
        self.admits_df = admits_df
        self.census_df = census_df
        self.daily_growth = daily_growth_helper(p.doubling_time)
        self.daily_growth_t = daily_growth_helper(doubling_time_t)


def sir(
    sy: float, iy: float, ry: float, so: float, io: float, ro: float, 
    betayy: float, betayo: float, betaoy: float, betaoo:float, gamma: float, 
    ny: float, no: float
) -> Tuple[float, float, float, float, float, float]:
    """The SIR model, one time step."""
    sy_n = (-betayy * sy * iy) + (-betayo * sy * io) + sy
    iy_n = (betayy * sy * iy + betayo * sy * io - gamma * iy) + iy
    ry_n = gamma * iy + ry
    so_n = (-betaoo * so * io) + (-betaoy * so * iy) + so
    io_n = (betaoo * so * io + betaoy * so * iy - gamma * io) + io
    ro_n = gamma * io + ro

    sy_n = 0.0 if sy_n < 0.0 else sy_n
    iy_n = 0.0 if iy_n < 0.0 else iy_n
    ry_n = 0.0 if ry_n < 0.0 else ry_n
    so_n = 0.0 if so_n < 0.0 else so_n
    io_n = 0.0 if io_n < 0.0 else io_n
    ro_n = 0.0 if ro_n < 0.0 else ro_n

    scaley = ny / (sy_n + iy_n + ry_n)
    scaleo = no / (so_n + io_n + ro_n)
    return (sy_n * scaley, iy_n * scaley, ry_n * scaley, 
            so_n * scaleo, io_n * scaleo, ro_n * scaleo)


def gen_sir(
    s: float, i: float, r: float, po: float, beta: float, omega: float, 
    gamma: float, n_days: int
) -> Generator[Tuple[float, float, float], None, None]:
    """Simulate SIR model forward in time yielding tuples."""
    s, i, r = (float(v) for v in (s, i, r))
    so, io, ro = (v * po for v in (s, i, r))
    sy, iy, ry = (v * (1.0 - po) for v in (s, i, r))
    ny = sy + iy + ry
    no = so + io + ro
    betayy = beta
    betayo = beta * omega 
    betaoy = beta * omega
    betaoo = beta * omega

    for d in range(n_days + 1):
        yield d, (sy + so), (iy + io), (ry + ro)
        sy, iy, ry, so, io, ro = sir(sy, iy, ry, so, io, ro, betayy, betayo, 
                                     betaoy, betaoo, gamma, ny, no)


def sim_sir_df(
    s: float, i: float, r: float, po: float, beta: float, omega: float, 
    gamma: float, n_days: int
) -> pd.DataFrame:
    """Simulate the SIR model forward in time."""
    return pd.DataFrame(
        data=gen_sir(s, i, r, po, beta, omega, gamma, n_days),
        columns=("day", "susceptible", "infected", "recovered"),
    )

def build_dispositions_df(
    sim_sir_df: pd.DataFrame,
    rates: Dict[str, float],
    market_share: float,
) -> pd.DataFrame:
    """Get dispositions of patients adjusted by rate and market_share."""
    patients = sim_sir_df.infected + sim_sir_df.recovered
    return pd.DataFrame({
        "day": sim_sir_df.day,
        **{
            key: patients * rate * market_share
            for key, rate in rates.items()
        }
    })


def build_admits_df(dispositions_df: pd.DataFrame) -> pd.DataFrame:
    """Build admits dataframe from dispositions."""
    admits_df = dispositions_df.iloc[:-1, :] - dispositions_df.shift(1)
    admits_df.day = dispositions_df.day
    return admits_df


def build_census_df(
    admits_df: pd.DataFrame,
    lengths_of_stay: Dict[str, int],
) -> pd.DataFrame:
    """ALOS for each disposition of COVID-19 case (total guesses)"""
    return pd.DataFrame({
        'day': admits_df.day,
        **{
            key: (
                admits_df[key].cumsum().iloc[:-los]
                - admits_df[key].cumsum().shift(los).fillna(0)
            ).apply(np.ceil)
            for key, los in lengths_of_stay.items()
        }
    })


def daily_growth_helper(doubling_time):
    """Calculates average daily growth rate from doubling time"""
    result = 0
    if doubling_time != 0:
        result = (np.power(2, 1.0 / doubling_time) - 1) * 100
    return result
