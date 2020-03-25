"""Models.

Changes affecting results or their presentation should also update
parameters.py `change_date`, so users can see when results have last
changed
"""

from __future__ import annotations

from typing import Dict, Generator, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from parameters import Parameters


class SimSirModel:

    def __init__(self, p: Parameters) -> SimSirModel:

        # Note: this should not be an integer.
        # We're appoximating infected from what we do know.
        # TODO market_share > 0, hosp_rate > 0
        self.infected = infected = (
            p.current_hospitalized / p.market_share / p.hospitalized.rate
        )

        self.detection_probability = (
            p.known_infected / infected if infected > 1.0e-7 else None
        )

        # TODO missing initial recovered value
        self.recovered = recovered = 0.0

        self.intrinsic_growth_rate = intrinsic_growth_rate = \
            (2.0 ** (1.0 / p.doubling_time) - 1.0) if p.doubling_time > 0.0 else 0.0

        self.gamma = gamma = 1.0 / p.recovery_days

        # Contact rate, beta
        self.beta = beta = (
            (intrinsic_growth_rate + gamma)
            / p.susceptible
            * (1.0 - p.relative_contact_rate)
        )  # {rate based on doubling time} / {initial susceptible}

        # r_t is r_0 after distancing
        self.r_t = beta / gamma * p.susceptible

        # Simplify equation to avoid division by zero:
        # self.r_naught = r_t / (1.0 - relative_contact_rate)
        self.r_naught = (intrinsic_growth_rate + gamma) / gamma

        # doubling time after distancing
        # TODO constrain values np.log2(...) > 0.0
        self.doubling_time_t = 1.0 / np.log2(
            beta * p.susceptible - gamma + 1)

        self.raw_df = raw_df = sim_sir_df(
            p.susceptible,
            infected,
            recovered,
            beta,
            gamma,
            p.n_days,
        )

        rates = {
            key: d.rate
            for key, d in p.dispositions.items()
        }

        lengths_of_stay = {
            key: d.length_of_stay
            for key, d in p.dispositions.items()
        }

        i_dict_v = get_dispositions(raw_df.infected, rates, p.market_share)
        r_dict_v = get_dispositions(raw_df.recovered, rates, p.market_share)

        self.dispositions = {
            key: value + r_dict_v[key]
            for key, value in i_dict_v.items()
        }

        self.dispositions_df = pd.DataFrame(self.dispositions)
        self.admits_df = admits_df = build_admits_df(p.n_days, self.dispositions)
        self.census_df = build_census_df(admits_df, lengths_of_stay)


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
    sy: float, iy: float, ry: float, so: float, io: float, ro: float, 
    betayy: float, betayo: float, betaoy: float, betaoo:float, gamma: float, 
    n_days: int
) -> Generator[Tuple[float, float, float], None, None]:
    """Simulate SIR model forward in time yielding tuples."""
    sy, iy, ry, so, io, ro = (float(v) for v in (sy, iy, ry, so, io, ro))
    ny = sy + iy + ry
    no = so + io + ro
    for d in range(n_days + 1):
        yield d, sy, iy, ry, so, io, ro
        sy, iy, ry, so, io, ro = sir(sy, iy, ry, so, io, ro, betayy, betayo, 
                                     betaoy, betaoo, gamma, ny, no)


def sim_sir_df(
    sy: float, iy: float, ry: float, so: float, io: float, ro: float, 
    betayy: float, betayo: float, betaoy: float, betaoo:float, gamma: float, 
    n_days
) -> pd.DataFrame:
    """Simulate the SIR model forward in time."""
    return pd.DataFrame(
        data=gen_sir(sy, iy, ry, so, io, ro, betayy, betayo, betaoy, betaoo, 
                     gamma, n_days),
        columns=("day", "susceptible", "infected", "recovered"),
    )


def get_dispositions(
    patients: np.ndarray,
    rates: Dict[str, float],
    market_share: float,
) -> Dict[str, np.ndarray]:
    """Get dispositions of patients adjusted by rate and market_share."""
    return {
        key: patients * rate * market_share
        for key, rate in rates.items()
    }


def build_admits_df(n_days, dispositions) -> pd.DataFrame:
    """Build admits dataframe from Parameters and Model."""
    days = np.arange(0, n_days + 1)
    projection = pd.DataFrame({
        "day": days,
        **dispositions,
    })
    # New cases
    admits_df = projection.iloc[:-1, :] - projection.shift(1)
    admits_df["day"] = range(admits_df.shape[0])
    return admits_df


def build_census_df(
    admits_df: pd.DataFrame, lengths_of_stay
) -> pd.DataFrame:
    """ALOS for each category of COVID-19 case (total guesses)"""
    n_days = np.shape(admits_df)[0]
    census_dict = {}
    for key, los in lengths_of_stay.items():
        census = (
            admits_df.cumsum().iloc[:-los, :]
            - admits_df.cumsum().shift(los).fillna(0)
        ).apply(np.ceil)
        census_dict[key] = census[key]

    census_df = pd.DataFrame(census_dict)
    census_df["day"] = census_df.index
    census_df = census_df[["day", *lengths_of_stay.keys()]]
    census_df = census_df.head(n_days)
    return census_df
