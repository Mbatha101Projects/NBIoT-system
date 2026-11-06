import pandas as pd
df = pd.read_csv("results/random_scenarios_results.csv")

df["win_naive"]  = df.adaptive_reward > df.naive_static_reward
df["win_best"]   = df.adaptive_reward > df.best_static_reward
df["gap_best"]   = df.adaptive_reward - df.best_static_reward

print("Win rate vs naive, per generator:")
print(df.groupby("generator_type").win_naive.mean().round(4))
print("Overall vs naive:", round(df.win_naive.mean(), 4))
print("Overall vs best static:", round(df.win_best.mean(), 4))
print("Worst single loss vs best static:", round(df.gap_best.min(), 4))
print(df.loc[df.gap_best.idxmin(), ["generator_type", "scenario_index",
      "adaptive_reward", "best_static_reward"]])