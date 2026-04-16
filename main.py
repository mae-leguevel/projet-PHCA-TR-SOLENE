import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import modules.River as r
import modules.Price as p
import modules.Conduite as c
import modules.PHCA_test as phca



def timescale_datetime(start, stop):
    # Datetime span by the hour
    datetime_index = pd.date_range(start=start, end=stop, freq='h')
    
    #Dataframe with Datetime index
    df = pd.DataFrame(index=datetime_index)
    
    # Reltaive time scale
    df['Temps relatif (h)'] = range(len(df))
    

    df.index.name = 'Datetime'
    
    return df



import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(df, reservoir_name="Reservoir1"):
    # On passe à 5 graphiques pour plus de clarté
    fig, (ax_q, ax_p, ax_thermo, ax_pres, ax_vol) = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    
    # --- 1. HYDRAULIQUE (Rivière) ---
    ax_q.plot(df['Temps relatif (h)'], df['Ariege_QmnJ'], label='Débit Rivière (QmnJ)', color='blue', alpha=0.5)
    ax_q.fill_between(df['Temps relatif (h)'], df['Ariege_Q_prelevable'], color='cyan', alpha=0.3, label='Débit Prélevable')
    ax_q.set_ylabel("Débit (m³/s)")
    ax_q.legend(loc='upper right')
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title(f"Analyse Complète du Système PHCA : {reservoir_name}")

    # --- 2. ÉCONOMIE & COMMANDE ---
    ax_p.plot(df['Temps relatif (h)'], df['Price_Price (EUR/MWhe)'], label='Prix Marché', color='orange', linewidth=2)
    ax_p.set_ylabel("Prix (€/MWh)")
    # On ajoute la commande sur le même axe (axe secondaire) pour voir la corrélation
    ax_cmd = ax_p.twinx()
    ax_cmd.step(df['Temps relatif (h)'], df['Price_Q_of'], where='post', color='purple', alpha=0.3, label='Cmd (1:P, -1:T)')
    ax_cmd.set_ylabel("Commande")
    ax_p.legend(loc='upper left')
    ax_p.grid(True, alpha=0.3)

    # --- 3. TEMPÉRATURES ---
    # On affiche T_air, T_eau et T_paroi si elles existent dans le DF
    if f'{reservoir_name}_T_air' in df.columns:
        ax_thermo.plot(df['Temps relatif (h)'], df[f'{reservoir_name}_T_air'] - 273.15, label='T° Air', color='red')
    # Ajoute ici T_eau et T_paroi si tu les as exportées dans process_results
    ax_thermo.set_ylabel("Température (°C)")
    ax_thermo.legend(loc='upper right')
    ax_thermo.grid(True, alpha=0.3)

    # --- 4. PRESSION ---
    if f'{reservoir_name}_Pression' in df.columns:
        # Conversion en bar pour plus de lisibilité (1 bar = 10^5 Pa)
        ax_pres.plot(df['Temps relatif (h)'], df[f'{reservoir_name}_Pression'] / 1e5, label='Pression Air', color='darkblue', linewidth=2)
    ax_pres.set_ylabel("Pression (bar)")
    ax_pres.legend(loc='upper right')
    ax_pres.grid(True, alpha=0.3)

    # --- 5. REMPLISSAGE (Volume en m³) ---
    if f'{reservoir_name}_z' in df.columns:
        # Calcul du volume : V = Surface * hauteur z
        # On suppose que tu as accès à self.A, sinon on le recalcule ou on l'extrait
        # Ici je cherche si 'A' est dispo, sinon on utilise une valeur générique ou z
        z = df[f'{reservoir_name}_z']
        # On affiche le volume
        ax_vol.fill_between(df['Temps relatif (h)'], z, color='blue', alpha=0.4, label='Niveau d\'eau (m)')
        ax_vol.set_ylabel("Niveau z (m)")
        
        # Axe secondaire pour le Volume en m3
        ax_m3 = ax_vol.twinx()
        # Calcul de la surface (si R=68)
        area = np.pi * 68**2 
        ax_m3.plot(df['Temps relatif (h)'], z * area, color='black', linestyle='--', alpha=0.0) # invisible juste pour l'échelle
        ax_m3.set_ylabel("Volume (m³)")
        
    ax_vol.set_xlabel("Temps relatif (heures)")
    ax_vol.legend(loc='upper right')
    ax_vol.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



####################################################
####################################################
"""         Test du modèle de PHCA complet       """
####################################################
####################################################


START = "2025-01-01"
STOP = "2025-01-08"

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



#Inner PHCA model
cuve = phca.PHCA(name="Reservoir1", H=68, R=68, z0=10, start=START, stop=STOP)
df_global = cuve.update(df, Q_p_max=25.0, Q_t_max=10.0)

print(df_global.columns)

plot_simulation_results(df)