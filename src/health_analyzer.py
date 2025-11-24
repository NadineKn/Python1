import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class HealthAnalyzer:
    """
    En klass som kan utföra någon del av analysen (t.ex. beräkna statistik eller rita grafer).

    Funktioner:
    - Räkna ut medel, median, min och max för: age, weight, height, systolic_bp, cholesterol.
    - Histogram över blodtryck.
    - Boxplot över vikt per kön.
    - Stapeldiagram över andelen rökare.
    """

    def __init__(self, df):
        self.df = df

    def desc(self):
        """
        Räkna ut medel, median, min och max för: age, weight, height, systolic_bp, cholesterol.
        """
        return (self.df[["age", "height", "weight", "systolic_bp", "cholesterol"]]
        .agg(['mean', 'median', 'min', 'max'])
        )

    def plot_bp_histogram(self):
        """
        Histogram över blodtryck.
        """
        fig, ax = plt.subplots()
        ax.hist(self.df["systolic_bp"], bins = 25, edgecolor = 'black')
        ax.set_title('Histogram över blodtryck')
        ax.set_xlabel('Systoliskt blodtryck (mmHg)')
        ax.set_ylabel('Antal personer')
        ax.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_weight_sex(self):
        """
        Boxplot över vikt per kön.
        """
        fig, ax = plt.subplots()
        self.df.boxplot(column="weight", by="sex", ax=ax)
        ax.set_title('Boxplot över vikt per kön')
        ax.set_xlabel('Kön')
        ax.set_ylabel('Vikt (kg)')
        ax.grid(True)
        plt.suptitle("")
        plt.tight_layout()
        plt.show()

    def smoker_counts(self):
        """
        Räkna ut andelen rökare.
        """
        return self.df["smoker"].value_counts(normalize=True)

    def plot_smoker(self):
        """
        Stapeldiagram över andelen rökare.
        """
        smoker_counts = self.smoker_counts()

        fig, ax = plt.subplots()
        ax.bar(smoker_counts.index, smoker_counts.values)
        ax.set_title('Andelen rökare')
        ax.set_xlabel('Rökare')
        ax.set_ylabel('Andel')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


    def plot_disease(self):
        """
        Stapeldiagram över sjukdomsförekomst per kön.
        """
        disease_counts = self.df.groupby("sex")["disease"].mean()

        fig, ax = plt.subplots()
        ax.bar(disease_counts.index, disease_counts.values)
        ax.set_title('Sjukdomsförekomst per kön (Female/Male)')
        ax.set_xlabel('Kön (Female/Male)')
        ax.set_ylabel('Andel')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


    def sim_disease(self, n=1000, seed=42):
        """
        Simulerar sjukdomsförekomst baserat på datasetets verkliga sannolikhet.
        n : Antal simulerade personer (default: 1000).
        seed : Slumpfrö för reproducerbarhet (default: 42).
        """

        # Andelen personer i datasetet som har sjukdomen
        disease_avg = self.df["disease"].mean()

        # Simulering
        np.random.seed(seed)
        sim = np.random.choice([0, 1], size=n, p=[1 - disease_avg, disease_avg])
        sim_avg = sim.mean()

        sim_result = {
            "real": float(disease_avg),
            "simulation": float(sim_avg),
            "difference": float(disease_avg - sim_avg)
        }

        return sim_result
    
    def print_sim_disease(self, n=1000, seed=42):
        """
        Skriver ut resultatet från simulerad sjukdomsförekomst.
        """
        result = self.sim_disease(n=n, seed=seed)

        print("Simulering av sjukdomsförekomst")
        print("----------------------------------")
        print(f"Andelen personer i datasetet som har sjukdomen:      {result['real']:.3f}")
        print(f"Andelen personer i simuleringen som har sjukdomen:   {result['simulation']:.3f}")
        print(f"Skillnad:                                            {result['difference']:.3f}")