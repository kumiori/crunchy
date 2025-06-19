import pandas as pd
# As per
# Maraghechi, S., Bosco, E., Suiker, A. S. J., and Hoefnagels, J. P. M. (2023).  Experimental characterisation of the local mechanical behaviour of cellulose fibres: an in-situ micro-profilometry approach, Cellulose, 30(7), 4225--4245

data = {
    "Fiber Type": ["Viscose Fiber", "Whatman Paper Fiber", "Aged Paper Fiber (1834)"],
    "Young's Modulus (GPa)": [8.1, 21.63, 15.0],
    "Std Dev (Young's Modulus) (GPa)": [0.6, 7.3, 2.1],
    "Strain at Fracture (-)": [0.160, 0.042, 0.034],
    "Std Dev (Strain at Fracture) (-)": [0.010, 0.008, 0.004],
    "Ultimate Tensile Strength (MPa)": [467, 621, 585],
    "Notes": [
        "Smooth surface; regular geometry; ductile failure.",
        "Irregular geometry; heterogeneous material properties; variations within and across fibers.",
        "Brittle fracture; embrittlement due to aging; limited twisting in local measurements possible.",
    ],
}

df = pd.DataFrame(data)

df["Young's Modulus Variation (± GPa)"] = ["±0.6", "±7.3", "±2.1"]
df["Strain at Fracture Variation (±)"] = ["±0.010", "±0.008", "±0.004"]
