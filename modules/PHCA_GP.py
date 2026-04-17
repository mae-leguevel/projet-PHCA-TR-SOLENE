import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy.interpolate as interpol
from base.components import BaseComponent
import CoolProp.CoolProp as CP

class PHCA(BaseComponent):    
    def __init__(self, name, H, R, z0, start, stop, P0=10e5, T_air0=293.15, physics_flags = None):
        super().__init__(name=name, start=start, stop=stop)
        
        
        # Geometry
        self.H, self.R, self.z0 = H, R, z0
        self.A = np.pi * R**2
        self.P0, self.T_air0 = P0, T_air0   #in Pa and K
        
        # Thermodynamics & Mass
        self.params = self._init_physical_constants()
        self.m_air = self._calculate_initial_air_mass()
        self.M_paroi = self._calculate_paroi_masse()

        self.physics = physics_flags if physics_flags is not None else {
        'Dalton': True, 'Magnus': True, 'Convection': True, 
        'Conduction': True, 'Spray': False, 'Variable_H': True
        }


    def _init_physical_constants(self):
        return {
            'r_air': 287.058, 'cv_air': 718.0, 'cp_air': 1005.0,
            'cp_eau': 4184.0, 'rho_eau': 1000.0, 'L_v': 2.26e6,
            'r_vap' : 461.5, 'cv_vap': 1410.0, 'k_evap': 1e-8, 
            'epais_p': 0.2, 'rho_p': 2400.0,
            'cp_p': 800.0, 'cond_p': 1.5, 'cp_vap': 1860.0,
            'T_ext': 293.15, 'T_in': 288.15,
            'h_aw_ref': 10.0, 'h_ap_ref': 5.0, 'h_wp_ref': 50.0,
            'P_critique': 200e6, 'T_critique': 773.15,
            'mu_air': 1.85e-5, 'k_air': 0.026, 'Pr_air': 0.71,          
            'd_goutte': 0.002, 'v_chute': 5.0, 'Q_spray_vol': 0.35
        }

    def _calculate_initial_air_mass(self):
        V_air0 = self.A * (self.H - self.z0)
        return (self.P0 * V_air0) / (self.params['r_air'] * self.T_air0)
    
    def _calculate_paroi_masse(self):
        return ((2 * np.pi * self.R * self.H) + (2 * self.A)) * self.params['epais_p'] * self.params['rho_p']

    
    def _get_inputs(self, df_global):
        """Calculate command fucntions by interpolation"""
        self.f_q_riv = interpol.interp1d(df_global['Relativ_time_h'] * 3600, 
                                       df_global['Q_prelevable'], 
                                       kind='linear', fill_value="extrapolate")
        self.f_cmd_prix = interpol.interp1d(df_global['Relativ_time_h'] * 3600, 
                                         df_global['Price_command'], 
                                         kind='previous', fill_value="extrapolate")

    def _calculate_heat_transfers(self, z, T_air, T_eau, T_paroi):
        """Heat transfer calculations by Newton law"""
        #Surfaces
        S_lat_air = 2 * np.pi * self.R * (self.H - z)      
        S_lat_eau = 2 * np.pi * self.R * z               
        S_fond = self.A                                    
        S_total_ext = (2 * np.pi * self.R * self.H) + self.A 

        #Heat transfers
        Q_air_paroi = self.params['h_ap_ref'] * S_lat_air * (T_air - T_paroi)
        Q_air_eau = self.params['h_aw_ref'] * self.A * (T_air - T_eau)
        Q_eau_paroi = self.params['h_wp_ref'] * (S_lat_eau + S_fond) * (T_eau - T_paroi)
        Q_pertes_ext = (self.params['cond_p'] / self.params['epais_p']) * S_total_ext * (T_paroi - self.params['T_ext'])

        return Q_air_paroi, Q_air_eau, Q_eau_paroi, Q_pertes_ext

    def _equations_differentielles(self, t, Y, Q_p_max, Q_t_max):
        z, T_air, T_eau, m_vap, T_p = Y
        
        # 1. Stratégie de commande
        q_riv = self.f_q_riv(t)
        cmd = self.f_cmd_prix(t)
        q_pump = min(Q_p_max, q_riv) if (cmd == 1 and z < 0.95 * self.H) else 0.0
        q_turb = Q_t_max if (cmd == -1 and z > 0.05 * self.H) else 0.0
        q_net = q_pump - q_turb 

        # 2. Géométrie dynamique
        V_air = max(self.A * (self.H - z), 1e-3)
        V_eau = max(self.A * z, 1.0)
        S_int = self.A
        S_ap = 2 * np.pi * self.R * (self.H - z) + self.A 
        S_wp = 2 * np.pi * self.R * z + self.A 
        S_p_tot = 2 * np.pi * self.R * self.H + 2 * self.A

        # 3. Thermodynamique (Loi de Dalton)
        P_air = (self.m_air * self.params['r_air'] * T_air) / V_air
        P_vap = (m_vap * self.params['r_vap'] * T_air) / V_air if self.physics.get('Dalton', True) else 0.0
        P_int = P_air + P_vap

        # 4. Évaporation (Magnus-Tetens)
        if self.physics.get('Magnus', True):
            T_eau_C = T_eau - 273.15
            P_sat_eau = 611.2 * np.exp((17.62 * T_eau_C) / (243.12 + T_eau_C))
            dm_vap_dt = self.params['k_evap'] * S_int * (P_sat_eau - P_vap)
            Q_latente = dm_vap_dt * self.params['L_v']
        else:
            dm_vap_dt, Q_latente = 0.0, 0.0

        # 5. Échanges thermiques (Convection)
        if self.physics.get('Convection', True):
            P_safe = max(P_int, 1e5)
            f_p = (P_safe / 1e5)**0.8 if self.physics.get('Variable_H', True) else 1.0
            h_aw = self.params['h_aw_ref'] * f_p
            h_ap = self.params['h_ap_ref'] * f_p
            h_wp = self.params['h_wp_ref'] * (1 + 0.01 * max(0, T_eau - 293.15))
            
            Q_aw = h_aw * S_int * (T_air - T_eau)
            Q_ap = h_ap * S_ap * (T_air - T_p)
            Q_wp = h_wp * S_wp * (T_eau - T_p)
        else:
            Q_aw = Q_ap = Q_wp = 0.0

        # 6. Pertes vers l'extérieur (Conduction)
        Q_cond = (self.params['cond_p'] / self.params['epais_p']) * S_p_tot * (T_p - self.params['T_ext']) if self.physics.get('Conduction', True) else 0.0

        # 7. Module Spray
        flux_spray = 0.0
        # On considère ici que le spray prend l'eau du réservoir (circuit fermé)
        # Donc pas d'impact sur dz_dt sauf si on pompe l'eau de l'extérieur directement en brumisation
        if self.physics.get('Spray', True) and abs(q_net) > 1e-3 and (self.H - z) > 1.0:
            vol_goutte = (np.pi * self.params['d_goutte']**3) / 6
            sur_goutte = np.pi * self.params['d_goutte']**2
            t_dr = (self.H - z) / self.params['v_chute'] 
            n_dr = (self.params['Q_spray_vol'] * t_dr) / vol_goutte 
            S_ech_spray = n_dr * sur_goutte
            
            rho_air_inst = max((self.m_air + m_vap) / V_air, 0.1)
            Re = (rho_air_inst * self.params['v_chute'] * self.params['d_goutte']) / self.params['mu_air']
            Nu = 2.0 + 0.6 * (Re**0.5) * (0.71**0.33) # Pr_air ~ 0.71
            h_spray = Nu * 0.026 / self.params['d_goutte'] # k_air ~ 0.026
            
            T_goutte = self.params['T_in'] if q_net > 0 else T_eau
            flux_spray = h_spray * S_ech_spray * (T_air - T_goutte)

        # 8. Calcul des dérivées
        dz_dt = (q_net - (dm_vap_dt / self.params['rho_eau'])) / self.A
        dV_air_dt = - self.A * dz_dt
        
        # dT_air/dt
        m_cv_total = (self.m_air * self.params['cv_air']) + (m_vap * self.params['cv_vap'])
        dT_air_dt = (- P_int * dV_air_dt - Q_aw - Q_ap - flux_spray) / m_cv_total
        
        # dT_eau/dt
        apport_froid_val = self.params['rho_eau'] * self.params['cp_eau'] * max(0, q_net) * (self.params['T_in'] - T_eau)
        dT_eau_dt = (apport_froid_val + Q_aw - Q_wp - Q_latente + flux_spray) / (self.params['rho_eau'] * V_eau * self.params['cp_eau'])
        
        # dT_p/dt
        dT_p_dt = (Q_ap + Q_wp - Q_cond) / (self.M_paroi * self.params['cp_p'])

        return [dz_dt, dT_air_dt, dT_eau_dt, dm_vap_dt, dT_p_dt]

    def update(self, df_global, Q_p_max=20.0, Q_t_max=5.0, T_eau_init=288.15, T_paroi_init=293.15, m_vap_init=1.0):
        """
        Résolution du système avec le vecteur d'état [z, T_air, T_eau, m_vap, T_p]
        """
        # Vecteur initial cohérent avec ta méthode _equations_differentielles
        # Y0 = [z, T_air, T_eau, m_vap, T_p]
        Y0 = [self.z0, self.T_air0, T_eau_init, m_vap_init, T_paroi_init]
        
        # Préparation des fonctions d'interpolation (stratégie)
        self._get_inputs(df_global)
        
        # Temps en secondes pour le solveur
        t_span = (0, df_global['Relativ_time_h'].max() * 3600)
        t_eval = df_global['Relativ_time_h'].values * 3600
        
        # Appel du solveur
        sol = solve_ivp(
            fun=lambda t, Y: self._equations_differentielles(t, Y, Q_p_max, Q_t_max),
            t_span=t_span, 
            y0=Y0, 
            t_eval=t_eval, 
            method='RK45', 
            max_step=900
        )
        return self._process_results(sol, df_global)

    def _process_results(self, sol, df_global):
        """
        Récupération des résultats et calcul de la pression (Dalton)
        """
        # Extraction des variables d'état résolues
        z = sol.y[0]
        T_air = sol.y[1]
        T_eau = sol.y[2]
        m_vap = sol.y[3] # Attention à l'ordre dans ton return d'équations diff
        T_paroi = sol.y[4]
        
        # Calcul de la géométrie pour la pression
        V_air = self.A * (self.H - z)
        
        # Recalcul de la pression selon la loi de Dalton (Air + Vapeur)
        # On utilise les constantes r_air et r_vap définies dans self.params
        P_air = (self.m_air * self.params['r_air'] * T_air) / V_air
        P_vap = (m_vap * self.params['r_vap'] * T_air) / V_air

        
        # Stockage dans le DataFrame global
        df_global['z'] = z
        df_global['T_air'] = T_air
        df_global['T_eau'] = T_eau
        df_global['T_paroi'] = T_paroi
        df_global['m_vap'] = m_vap
        df_global['Pression'] = P_air + P_vap # Pression totale
        
        # Volume d'eau en m3 pour ton graphique
        df_global['V_eau_m3'] = self.A * z

        dz = df_global['z'].diff()
        dt = df_global['Relativ_time_h'].diff() * 3600   
        q = self.A * (dz / dt)
        df_global['Q_in'] = q.clip(lower=0)
        df_global['Q_in'] = df_global['Q_in'].fillna(0)
        df_global['Q_out'] = q.clip(upper=0).abs()
        df_global['Q_out'] = df_global['Q_out'].fillna(0)
    
        return df_global
    def fetch_data(self):
        """No need"""
        pass