import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import modules.River as r
import modules.Price as p
import modules.Conduite as c
import modules.PHCA_GR as phca_gr
import modules.PHCA_GP as phca_gp



def timescale_datetime(start, stop):
    # Datetime span by the hour
    datetime_index = pd.date_range(start=start, end=stop, freq='h')
    
    #Dataframe with Datetime index
    df = pd.DataFrame(index=datetime_index)
    
    # Reltaive time scale
    df['Relativ_time_h'] = range(len(df))
    

    df.index.name = 'Datetime'
    
    return df



def plot_simulation_results(df_list, names_list, flags=None):
    if flags is None:
        flags = {
            'hydro': True, 'eco': True, 'temp': True, 
            'pres': True, 'vol': True, 'phca_q': True, 
            'power': True
        }
    
    active_axes = [k for k, v in flags.items() if v]
    n_axes = len(active_axes)
    
    fig, axes = plt.subplots(n_axes, 1, figsize=(12, 3 * n_axes), sharex=True)
    if n_axes == 1: axes = [axes]
    
    ax_map = {name: axes[i] for i, name in enumerate(active_axes)}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    linestyles = ['-', '--', ':', '-.']

    for i, (df, name) in enumerate(zip(df_list, names_list)):
        c = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        t = df['Relativ_time_h']

        # --- 1. HYDRAULIQUE ---
        if flags.get('hydro') and 'Q_prelevable' in df.columns:
            ax = ax_map['hydro']
            ax.plot(t, df['Q_prelevable'], label=f'Q_prel ({name})', color=c, linestyle=ls, alpha=0.8)
            ax.set_ylabel("Débit Rivière (m³/s)")

        # --- 2. ÉCONOMIE ---
        if flags.get('eco') and 'Price_command' in df.columns:
            ax = ax_map['eco']
            ax.step(t, df['Price_command'], where='post', color=c, linestyle=ls, label=f'Cmd ({name})', alpha=0.7)
            ax.set_ylabel("Commande")

        # --- 3. TEMPÉRATURES ---
        if flags.get('temp') and 'T_air' in df.columns:
            ax = ax_map['temp']
            ax.plot(t, df['T_air'] - 273.15, color=c, linestyle=ls, label=f'Air ({name})', alpha=0.9)
            if i == 0: 
                ax.twin = ax.twinx()
                ax.twin.set_ylabel("T° Eau (°C)")
            if 'T_eau' in df.columns:
                ax.twin.plot(t, df['T_eau'] - 273.15, color=c, linestyle=':', alpha=0.5, label=f'Eau ({name})')

        # --- 4. PRESSION ---
        if flags.get('pres') and 'Pression' in df.columns:
            ax = ax_map['pres']
            ax.plot(t, df['Pression'] / 1e5, color=c, linestyle=ls, label=f'P ({name})', alpha=0.8)
            ax.set_ylabel("Pression (bar)")

        # --- 5. VOLUME (z) ---
        if flags.get('vol') and 'z' in df.columns:
            ax = ax_map['vol']
            ax.plot(t, df['z'], color=c, linestyle=ls, label=f'z ({name})', alpha=0.8)
            ax.set_ylabel("Niveau z (m)")

        # --- 6. DÉBITS PHCA ---
        if flags.get('phca_q'):
            ax = ax_map['phca_q']
            if 'Q_in' in df.columns:
                ax.plot(t, df['Q_in'], label=f'Q_pump ({name})', color=c, linestyle='-', alpha=0.8)
            if 'Q_out' in df.columns:
                ax.plot(t, df['Q_out'], label=f'Q_turb ({name})', color=c, linestyle='--', alpha=0.5)
            ax.set_ylabel("Débits PHCA (m³/s)")
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        # --- 7. PUISSANCES ÉLECTRIQUES ---
        if flags.get('power'):
            ax = ax_map['power']
            if 'P_elec_in_MW' in df.columns:
                # Puissance consommée (positive pour la visibilité)
                ax.plot(t, df['P_elec_in_MW'], label=f'P_consommée ({name})', color=c, linestyle='-', linewidth=2)
            if 'P_elec_out_MW' in df.columns:
                # Puissance produite
                ax.plot(t, df['P_elec_out_MW'], label=f'P_produite ({name})', color=c, linestyle='--', linewidth=2)
            ax.set_ylabel("Puissance (MW)")
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

    # Finalisation propre
    for ax_name, ax in ax_map.items():
        handles, labels = ax.get_legend_handles_labels()
        
        if ax_name == 'temp' and hasattr(ax, 'twin'):
            h2, l2 = ax.twin.get_legend_handles_labels()
            handles += h2
            labels += l2

        if handles:
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
        
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.xlabel("Temps relatif (heures)")
    plt.tight_layout()
    plt.subplots_adjust(right=0.82) 
    plt.show()

def compute_powers(df, eta_pump=0.85, eta_turb=0.85):
    
    rho = 1e3
    g = 9.81

    
    H_pompage_pa = df['Pression'] + df['IN_DeltaP'] + (rho * g * df['z'])

    df['P_elec_in_MW'] = (df['Q_in'] * H_pompage_pa) / eta_pump / 1e6

    
    H_turbinage_pa = df['Pression'] + (rho * g * df['z']) - df['OUT_DeltaP']
    H_turbinage_pa = H_turbinage_pa.clip(lower=0)
    
    df['P_elec_out_MW'] = (df['Q_out'] * H_turbinage_pa) * eta_turb / 1e6

    df['P_net_MW'] = df['P_elec_out_MW'] - df['P_elec_in_MW']
    
    return df
    



####################################################
####################################################
"""         Test du modèle de PHCA complet       """
####################################################
####################################################


START = "2025-01-01"
STOP = "2025-01-5"

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
riviere.apply_strategy(method='simplified', source_col="QmnJ", param=5.0)
df = riviere.update(df)

#Data Price
PATH = "../data/France.csv"
#initialization
price = p.Price(path=PATH, start=START, stop=STOP)
price.fetch_data()
price.compute_thresholds()
price.apply_strategy(method='simplified')
df = price.update(df)

#Conduite losses
#Geometry
D = 1
L = 200
DELTA_Z = 200
COMPLEXITY = 2
#initialization
conduiteLosses = c.Conduit(name="IN", D=D, L=L, Delta_z=DELTA_Z, start=START, stop=STOP, complexity=COMPLEXITY)
df = conduiteLosses.update(df, q_name= "Q_prelevable" )

"""
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
#Inner PHCA model
cuve = phca_gr.PHCA(name="Reservoir1", H=10, R=100, z0=6, start=START, stop=STOP)
df_global1 = cuve.update(df1, Q_p_max=10.0, Q_t_max=2.0)
flags = {
        'Dalton': True, 'Magnus': True, 'Convection': True, 
        'Conduction': True, 'Spray': False, 'Variable_H': True
    }
cuve2 = phca_gp.PHCA(name="Reservoir2", H=10, R=100, z0=6, start=START, stop=STOP, physics_flags=flags)

df_global2 = cuve2.update(df2, Q_p_max=10.0, Q_t_max=2.0)

flags = {
        'Dalton': True, 'Magnus': True, 'Convection': True, 
        'Conduction': True, 'Spray': True, 'Variable_H': True
    }
cuve3 = phca_gp.PHCA(name="Reservoir2", H=10, R=100, z0=6, start=START, stop=STOP, physics_flags=flags)

df_global3 = cuve3.update(df3, Q_p_max=10.0, Q_t_max=2.0)

plot_simulation_results(df_list=[df_global1,df_global2, df_global3],
                        names_list=['GR','GP','GP_Spray'], 
                        flags={'hydro': False, 'eco': False, 'temp': True, 'pres': True, 'vol': True, 'phca_q' : True})
"""


cuve = phca_gr.PHCA(name="Reservoir1", H=10, R=100, z0=6, start=START, stop=STOP)
df_global = cuve.update(df, Q_p_max=10.0, Q_t_max=2.0)

conduiteLosses = c.Conduit(name="OUT", D=D, L=L, Delta_z=-DELTA_Z, start=START, stop=STOP, complexity=COMPLEXITY)
df_global = conduiteLosses.update(df_global, q_name= "Q_out")

df_global = compute_powers(df_global)
plot_simulation_results(df_list=[df_global],
                        names_list=['GR'], 
                        flags={'hydro': False, 'eco': False, 'temp': True, 'pres': True, 'vol': True, 'phca_q' : True, 'power' : True})



