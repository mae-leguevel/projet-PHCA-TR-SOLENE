import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import modules.River as r
import modules.Price as p
import modules.Conduite as c



def timescale_datetime(start, stop):
    # Datetime span by the hour
    datetime_index = pd.date_range(start=start, end=stop, freq='h')
    
    #Dataframe with Datetime index
    df = pd.DataFrame(index=datetime_index)
    
    # Reltaive time scale
    df['Temps relatif (h)'] = range(len(df))
    

    df.index.name = 'Datetime'
    
    return df



def plot_simulation_results(df):
    fig, (ax_q, ax_p, ax_cmd) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Hydro
    ax_q.plot(df['Temps relatif (h)'], df['Ariege_QmnJ'], label='Débit Rivière (QmnJ)', color='blue', alpha=0.5)
    ax_q.fill_between(df['Temps relatif (h)'], df['Ariege_Q_prelevable'], color='cyan', alpha=0.3, label='Débit Prélevable')
    ax_q.set_ylabel("Débit (m³/s)")
    ax_q.legend(loc='upper right')
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title("Analyse Hydraulique et Économique du Système")

    # 2. Eco
    ax_p.plot(df['Temps relatif (h)'], df['Price_Price (EUR/MWhe)'], label='Prix Marché', color='orange', linewidth=2)
    ax_p.plot(df['Temps relatif (h)'], df['Price_Mean_Daily'], '--', label='Moyenne Journalière', color='green', alpha=0.7)
    ax_p.plot(df['Temps relatif (h)'], df['Price_Peak_Daily'], '--', label='Seuil Peak', color='red', alpha=0.7)
    ax_p.set_ylabel("Prix (€/MWh)")
    ax_p.legend(loc='upper right')
    ax_p.grid(True, alpha=0.3)

    # 3. Commande
    ax_cmd.step(df['Temps relatif (h)'], df['Price_Q_of'], where='post', color='purple', label='Commande (1:Pump, -1:Turb)')
    ax_cmd.axhline(0, color='black', linewidth=1)
    ax_cmd.set_ylabel("État Système")
    ax_cmd.set_ylim(-1.5, 1.5)
    ax_cmd.set_xlabel("Temps relatif (heures)")
    ax_cmd.legend(loc='upper right')
    ax_cmd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



####################################################
####################################################
"""         Test du modèle de PHCA complet       """
####################################################
####################################################


START = "2026-01-01"
STOP = "2026-01-20"

#Global DF initialization
df = timescale_datetime(START, STOP)


#Data River
#caracterization
CODE = "O1372510"
NAME = "Ariege"
TYPES = ["QmnJ"]
#initialization
riviere = r.River(code=CODE, name=NAME, start=START, stop=STOP)
riviere.fetch_data(q_types = TYPES, include_temp=False)
riviere.apply_strategy(method='percentage', source_col="QmnJ", param=0.1)
df = riviere.update(df)

#Data Price
PATH = "../data/France.csv"
#initialization
price = p.Price(path=PATH, start=START, stop=STOP)
price.fetch_data()
price.compute_thresholds()
price.apply_strategy(method='threshold')
df = price.update(df)

#Conduite losses
#Geometry
D = 1
L = 200
DELTA_Z = 200
COMPLEXITY = 2
#initialization
conduiteLosses = c.Conduit(name="IN", D=D, L=L, Delta_z=DELTA_Z, start=START, stop=STOP, complexity=COMPLEXITY)
df = conduiteLosses.update(df, q_name= NAME+"_Q_prelevable" )
plot_simulation_results(df)

#Inner PHCA model
#geometry
H = 25
R = 10

#initial conditions
Z0 = 2
P0 = 10e5
T_AIR = 293.15
T_EAU = 293.15

