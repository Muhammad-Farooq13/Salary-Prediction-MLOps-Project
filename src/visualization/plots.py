from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.load_data import load_raw_data, preprocess


def scatter_experience_salary(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    ax = sns.scatterplot(data=df, x="YearsExperience", y="Salary")
    ax.set_title("Years of Experience vs. Salary")
    plt.show()


def main() -> None:
    df = load_raw_data(Path("data/raw/Salary_Data.csv"))
    df = preprocess(df)
    scatter_experience_salary(df)


if __name__ == "__main__":
    main()
