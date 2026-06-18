"""H2 cDFT — COF-333-CoCl2, Morse(Co) + LJ(rest), 298 K, 0-700 bar."""
from _h2cof_runner import run_isotherm
if __name__ == "__main__":
    run_isotherm("COF-333-CoCl2", mode="LJ-Morse")
