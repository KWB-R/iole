

if __name__ == "__main__":
    leaks = pd.read_csv(r"data\artificial_leakages\LEAK_PATTERNS_SENSITIVITY.csv", index_col=0)
    leaks.index = pd.to_timedelta(leaks.index)