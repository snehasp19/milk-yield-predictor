import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

parity = np.random.randint(1, 6, n)
days_in_milk = np.random.randint(1, 306, n)
bcs = np.round(np.random.uniform(2.0, 4.5, n), 1)
feed_intake = np.round(np.random.uniform(10, 30, n), 1)

milk_yield = (
    0.8 * parity
    + 0.15 * feed_intake
    + 1.5 * bcs
    + 0.02 * days_in_milk
    - 0.00005 * (days_in_milk ** 2)
    + np.random.normal(0, 1.5, n)
)

milk_yield = np.round(milk_yield, 2)

df = pd.DataFrame({
    "Parity": parity,
    "Days_in_Milk": days_in_milk,
    "BCS": bcs,
    "Feed_Intake": feed_intake,
    "Milk_Yield": milk_yield
})

df.to_csv("milk_yield.csv", index=False)

print("Dataset generated successfully!")