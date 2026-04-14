import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class PHCA_Model:
    """
    Classe modélisant le système PHCA
    Séparation des paramètres physiques de la résolution algorithmique.
    Controle de débit basé sur une stratégie de prix/débit du cours d'eau.
    """
    
    def __init__(self, H=25.0, R=10.0, z0=2.0, P0=10e5, T_air0=293.15, T_eau0=293.15, Q_out= 5.0,
                 Dalton=True, Magnus=True, Convection=True, Conduction=True, Variable_H=True, Spray=True):

        # --- CONFIGURATION DE LA PHYSIQUE ---
        self.active_physics = {
            'Dalton': Dalton,        
            'Magnus': Magnus,        
            'Convection': Convection,  
            'Conduction': Conduction,  
            'Variable_H': Variable_H,
            'Spray': Spray  
        }    

        # --- CONSTANTES PHYSIQUES ---
        self.r_air = 287.058 # J/(kg·K) constante spécifique de l'air
        self.cv_air = 718.0 # J/(kg·K) capacité calorifique à volume constant de l'air
        self.cv_vap = 1410.0 # J/(kg·K) capacité calorifique à volume constant de la vapeur
        self.cp_air = 1005.0 # J/(kg·K) capacité calorifique à pression constante de l'air
        self.cp_eau = 4184.0 # J/(kg·K) capacité calorifique de l'eau
        self.rho_eau = 1000.0 # kg/m³ masse volumique de l'eau
        self.r_vap = 461.5 # J/(kg·K) constante spécifique de la vapeur
        self.L_v = 2.26e6 # J/kg chaleur latente de vaporisation
        self.k_evap = 1e-8 # kg/(s·m²·Pa) coefficient de transfert de chaleur par évaporation
        self.epais_p = 0.2 # m épaisseur de la paroi
        self.rho_p = 2400.0 # kg/m³ masse volumique de la paroi
        self.cp_p = 800.0 # J/(kg·K) capacité calorifique de la paroi
        self.cond_p = 1.5 # W/(m·K) conductivité thermique de la paroi

        # --- PPT DU SPRAY ---
        self.mu_air = 1.85e-5   # Pa·s viscosité dynamique de l'air
        self.k_air = 0.026       # W/(m·K) conductivité thermique de l'air
        self.Pr_air = 0.71       # nombre de Prandtl de l'air
        self.d_goutte = 0.002    # m diamètre des gouttes de spray
        self.v_chute = 5.0       # m/s vitesse de chute des gouttes
        self.Q_spray_vol = 0.35   # m³/s débit de spray

        # --- PARAMÈTRES THERMIQUES ET LIMITES ---
        self.T_ext = 293.15        
        self.T_in = 288.15         
        self.h_aw_ref, self.h_ap_ref, self.h_wp_ref = 10.0, 5.0, 50.0
        self.P_critique, self.T_critique = 200e6, 773.15     

        # --- GÉOMÉTRIE ET ÉTAT INITIAL ---
        self.H, self.R = H, R
        self.A = np.pi * R**2
        self.z0, self.T_air0, self.T_eau0, self.m_vap0 = z0, T_air0, T_eau0, 0.0
        
        # Calcul de la masse d'air initiale (qui reste constante)
        V_air0 = self.A * (self.H - z0)   
        self.m_air = (P0 * V_air0) / (self.r_air * T_air0) 

        self.Q_fleuve = None
        self.strategie_fleuve = None
        self.fonction_prix = None
        self.strategie_prix = None
        self.Q_out = Q_out


    def debit_cours_eau(self, t, Q, strategie = lambda Q,t: Q):
        """
        Débit admissible du cours d'eau basé sur une stratégie.
        """
        self.Q_fleuve = interp1d(t, Q, kind='previous', bounds_error=False, fill_value=(Q[0], Q[-1]))
        self.strategie_fleuve = strategie
        

    def prix(self, temps_secondes, prix, strategie = lambda p,q,Q_out: q if p < 40 else Q_out if p > 65 else 0):
        """
        Activation de la pompe et de la turbine basé sur une stratégie de prix.
        """
        self.fonction_prix = interp1d(temps_secondes, prix, kind='previous', bounds_error=False, fill_value=(prix[0], prix[-1]))
        self.strategie_prix = strategie
        

    # ==========================================
    # 1. PARTIE PHYSIQUE
    # ==========================================
    def equations_differentielles(self, t, Y):
        z, T_air, T_eau, m_vap, T_p = Y
        z = max(Y[0], 1e-3) 
        T_air = max(Y[1], 10.0) 
        T_eau = max(Y[2], 273.15) 
        m_vap = max(Y[3], 0.0)
        T_p = Y[4]

        Q_admissible = self.strategie_fleuve(self.Q_fleuve(t))
        Q_effectif = self.strategie_prix(self.fonction_prix(t), Q_admissible, self.Q_out)


        # --- Géométrie ---
        V_air = self.A * (self.H - z)
        V_eau = max(self.A * z, 1.0)
        S_int, S_ap, S_wp = self.A, 2*np.pi*self.R*(self.H-z)+self.A, 2*np.pi*self.R*z+self.A
        S_p_tot = 2 * np.pi * self.R * self.H + 2 * self.A

        # --- Loi de Dalton ---
        P_air = (self.m_air * self.r_air * T_air) / V_air
        P_vap = (m_vap * self.r_vap * T_air) / V_air if self.active_physics['Dalton'] else 0.0
        P_int = P_air + P_vap

        # --- Loi de Magnus-Tetens ---
        if self.active_physics['Magnus']:
            T_eau_C = T_eau - 273.15
            P_sat_eau = 611.2 * np.exp((17.62 * T_eau_C) / (243.12 + T_eau_C))
            dm_vap_dt = self.k_evap * S_int * (P_sat_eau - P_vap)
            Q_latente = dm_vap_dt * self.L_v
        else:
            dm_vap_dt, Q_latente = 0.0, 0.0

        # --- Convection ---
        if self.active_physics['Convection']:
            f_p = (P_int / 1e5)**0.8 if self.active_physics['Variable_H'] else 1.0
            h_aw, h_ap = self.h_aw_ref * f_p, self.h_ap_ref * f_p
            h_wp = self.h_wp_ref * (1 + 0.01 * max(0, T_eau - 293.15))
            
            Q_aw, Q_ap, Q_wp = h_aw*S_int*(T_air-T_eau), h_ap*S_ap*(T_air-T_p), h_wp*S_wp*(T_eau-T_p)
        else:
            Q_aw = Q_ap = Q_wp = 0.0

        # --- Conduction ---
        Q_cond = (self.cond_p/self.epais_p)*S_p_tot*(T_p-self.T_ext) if self.active_physics['Conduction'] else 0.0

        # --- MODULE SPRAY ---
        flux_spray = 0.0
        qe_spray_ext = 0.0

        if self.active_physics['Spray'] and Q_effectif != 0 and (self.H - z) > 1.0:
            vol_goutte = (np.pi * self.d_goutte**3) / 6
            sur_goutte = np.pi * self.d_goutte**2
            t_dr = (self.H - z) / self.v_chute 
            
            n_dr = (self.Q_spray_vol * t_dr) / vol_goutte 
            S_ech_spray = n_dr * sur_goutte
            
            # Sécurité sur le Reynolds
            rho_air_inst = max(self.m_air / V_air, 0.1)
            Re = (rho_air_inst * self.v_chute * self.d_goutte) / self.mu_air
            Nu = 2.0 + 0.6 * (Re**0.5) * (self.Pr_air**(1/3))
            h_spray = Nu * self.k_air / self.d_goutte
            
            T_goutte = self.T_in if Q_effectif > 0 else T_eau
            flux_spray = h_spray * S_ech_spray * (T_air - T_goutte)
            
            if Q_effectif > 0:
                qe_spray_ext = self.Q_spray_vol

        # --- dérivées ---
        dz_dt = (Q_effectif + qe_spray_ext - (dm_vap_dt / self.rho_eau)) / self.A
        dV_air_dt = - self.A * dz_dt
        
        m_cv_total = (self.m_air * self.cv_air) + (m_vap * self.cv_vap)
        dT_air_dt = (- P_int * dV_air_dt - Q_aw - Q_ap - flux_spray) / m_cv_total
        
        Apport_froid = self.rho_eau * self.cp_eau * Q_effectif * (self.T_in - T_eau)
        dT_eau_dt = (Apport_froid + Q_aw - Q_wp - Q_latente + flux_spray) / (self.rho_eau * V_eau * self.cp_eau)
        
        M_p = S_p_tot * self.epais_p * self.rho_p
        dT_p_dt = (Q_ap + Q_wp - Q_cond) / (M_p * self.cp_p)

        return [dz_dt, dT_air_dt, dT_eau_dt, dm_vap_dt, dT_p_dt]

    # ==========================================
    # 2. PARTIE ALGORYTHMIQUE 
    # ==========================================
    def executer_simulation(self, Q_e_max, Q_s_max, t_end, nb_points=1000):
        
        # --- Sécurités SciPy (Événements) ---
        def event_pression(t, Y):
            z, T_air, _, m_vap, _ = Y 
            V_air = self.A * (self.H - z)
            P_int = (self.m_air * self.r_air * T_air) / V_air + (m_vap * self.r_vap * T_air) / V_air
            return self.P_critique - P_int
        event_pression.terminal = True
        
        def event_T_air(t, Y): return self.T_critique - Y[1]
        event_T_air.terminal = True
        
        def event_T_eau(t, Y): return self.T_critique - Y[2]
        event_T_eau.terminal = True

        # --- Résolution ---
        Y0 = [self.z0, self.T_air0, self.T_eau0, self.m_vap0, self.T_ext]
        temps_eval = np.linspace(0, t_end, nb_points)
        
        fonction_solveur = lambda t, Y: self.equations_differentielles(t, Y)
        
        solution = solve_ivp(
            fun=fonction_solveur, 
            t_span=(0, t_end), 
            y0=Y0, 
            method='RK45',          
            t_eval=temps_eval,      
            events=[event_pression, event_T_air, event_T_eau],
            rtol=1e-5, atol=1e-8,
            max_step=900 
        )
        
        return self._traiter_resultats(solution, Q_e_max, Q_s_max)

    # ==========================================
    # 3. TRAITEMENT DES RÉSULTATS POST-SIMULATION
    # ==========================================
    def _traiter_resultats(self, solution, Q_e_max, Q_s_max):
        t_sim = solution.t
        z_sim = solution.y[0]
        Ta_sim = solution.y[1] - 273.15 
        Te_sim = solution.y[2] - 273.15 
        m_vap_sim = solution.y[3]
        
        prix_sim = self.fonction_prix(t_sim)
        
        Q_e_hist = np.zeros_like(t_sim)
        Q_s_hist = np.zeros_like(t_sim)
        
        for i in range(len(t_sim)):
            p = prix_sim[i]
            z = z_sim[i]
            
            qe, qs = 0.0, 0.0
            if p < self.seuil_bas: qe = Q_e_max
            elif p > self.seuil_haut: qs = Q_s_max
                
            if z >= 0.9 * self.H: qe = 0.0
            if z <= 0.1 * self.H: qs = 0.0
                
            Q_e_hist[i] = qe
            Q_s_hist[i] = qs
        
        # Loi de Dalton a posteriori
        P_air_sim = (self.m_air * self.r_air * solution.y[1]) / (self.A * (self.H - z_sim))
        P_vap_sim = (m_vap_sim * self.r_vap * solution.y[1]) / (self.A * (self.H - z_sim))
        P_sim = P_air_sim + P_vap_sim
        
        # Messages d'état
        if solution.status == 1:
            print(f"Simulation interrompue par une contrainte à t={t_sim[-1]:.1f}s")
        elif solution.status == 0:
            print("Simulation terminée avec succès.")
            
        return {
            'temps': t_sim, 'z': z_sim, 'P_int': P_sim, 
            'T_air': Ta_sim, 'T_eau': Te_sim, 'm_vap': m_vap_sim,
            'prix': prix_sim, 'Q_e': Q_e_hist, 'Q_s': Q_s_hist 
        }

# ==========================================
# UTILISATION ET AFFICHAGE
# ==========================================
if __name__ == "__main__":
    
    df = pd.concat([
        dataprocess.load_data_infraday(dataprocess.path28)[['plot_time','Price (EUR/MWh)']],
        dataprocess.load_data_infraday(dataprocess.path30)[['plot_time','Price (EUR/MWh)']]
    ],)
    
    prix = df['Price (EUR/MWh)'].to_numpy()

    pas_de_temps_heures = 0.25 
    nb_points = len(prix)
    t_end_simulation = nb_points * pas_de_temps_heures * 3600
    
    temps = np.linspace(0, t_end_simulation, nb_points)

    modele = PHCA_Model(H=68.0, R=68.0, z0=10.0, P0=10e5, Spray=False)
    modele.strategie_prix(temps, prix, seuil_bas=40.0, seuil_haut=65.0)
 
    # Appel correct avec les bons noms de variables
    resultats = modele.executer_simulation(Q_e_max=20.0, Q_s_max=5.0, t_end=t_end_simulation)
    
    t_h = resultats['temps'] / 3600
    
    fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    # 1. Activation Pompe et Turbine
    axs[0].plot(t_h, resultats['Q_e'], 'b-', label="Pompe (Q_e) - Remplissage", linewidth=2)
    axs[0].plot(t_h, -resultats['Q_s'], 'r-', label="Turbine (Q_s) - Vidage", linewidth=2) 
    axs[0].set_ylabel("Débit (m³/s)")
    axs[0].set_title("1. Activation des machines (La turbine est affichée en négatif pour plus de clarté)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    # 2. Prix de l'électricité
    axs[1].plot(t_h, resultats['prix'], 'k-', label="Prix Marché")
    axs[1].axhline(y=modele.seuil_haut, color='r', linestyle='--', label=f"Seuil Turbine ({modele.seuil_haut}€)")
    axs[1].axhline(y=modele.seuil_bas, color='b', linestyle='--', label=f"Seuil Pompe ({modele.seuil_bas}€)")
    axs[1].set_ylabel("Euros / MWh")
    axs[1].set_title("2. Prix de l'électricité et stratégie d'arbitrage")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    # 3. Pression interne
    axs[2].plot(t_h, resultats['P_int'] / 1e5, 'purple', linewidth=2)
    axs[2].set_ylabel("Pression (bar)")
    axs[2].set_title("3. Pression de l'air dans la cuve")
    axs[2].grid(True)

    # 4. Températures
    axs[3].plot(t_h, resultats['T_air'], 'orange', label="T_air (Gaz)")
    axs[3].plot(t_h, resultats['T_eau'], 'cyan', label="T_eau (Liquide)")
    axs[3].set_ylabel("Température (°C)")
    axs[3].set_title("4. Évolution thermique")
    axs[3].legend(loc="upper right")
    axs[3].grid(True)

    # 5. Fraction de vapeur
    axs[4].plot(t_h, resultats['m_vap']/(resultats['m_vap'] + resultats['P_int']/(287.058*resultats['P_int'])), 'b-', linewidth=2)
    axs[4].axhline(y=25.0*0.9, color='red', linestyle=':', label="Sécurité Haute (90%)")
    axs[4].axhline(y=25.0*0.1, color='orange', linestyle=':', label="Sécurité Basse (10%)")
    axs[4].set_ylabel("Fraction de vapeur")
    axs[4].set_xlabel("Temps (heures)")
    axs[4].set_title("5. Fraction de vapeur")
    axs[4].legend(loc="lower right")
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()