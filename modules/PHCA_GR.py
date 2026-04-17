import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy.interpolate as interpol
from base.components import BaseComponent
import CoolProp.CoolProp as CP

class PHCA(BaseComponent):    
    def __init__(self, name, H, R, z0, start, stop, P0=10e5, T_air0=293.15):
        super().__init__(name=name, start=start, stop=stop)
        
        # Geometry
        self.H, self.R, self.z0 = H, R, z0
        self.A = np.pi * R**2
        self.P0, self.T_air0 = P0, T_air0   #in Pa and K
        
        # Thermodynamics & Mass
        self.params = self._init_physical_constants()
        self.m_air = self._calculate_initial_air_mass()
        self.M_paroi = self._calculate_paroi_masse()

        # Lookup Tables
        rho_min, rho_max, u_min, u_max = 1.0, 120.0, 200000, 450000 #in kg.m^-3 & J.kg^-1
        self.look_P, self.look_T = self.generate_thermo_lookup(1.0, 120.0, 200000, 450000, n=50)


    def _init_physical_constants(self):
        return {
            'r_air': 287.058, 'cv_air': 718.0, 'cp_air': 1005.0,
            'cp_eau': 4184.0, 'rho_eau': 1000.0, 'L_v': 2.26e6,
            'k_evap': 1e-8, 'epais_p': 0.2, 'rho_p': 2400.0,
            'cp_p': 800.0, 'cond_p': 1.5, 'cp_vap': 1860.0,
            'T_ext': 293.15, 'T_in': 288.15,
            'h_aw_ref': 10.0, 'h_ap_ref': 5.0, 'h_wp_ref': 50.0,
            'P_critique': 200e6, 'T_critique': 773.15
        }

    def _calculate_initial_air_mass(self):
        V_air0 = self.A * (self.H - self.z0)
        return (self.P0 * V_air0) / (self.params['r_air'] * self.T_air0)
    
    def _calculate_paroi_masse(self):
        return ((2 * np.pi * self.R * self.H) + (2 * self.A)) * self.params['epais_p'] * self.params['rho_p']

    @staticmethod
    def generate_thermo_lookup(rho_min, rho_max, u_min, u_max, n=50):
        """Compute P(rho, u) & T(rho, u) grids by interpolation"""
        rho_axis = np.linspace(rho_min, rho_max, n)
        u_axis = np.linspace(u_min, u_max, n)

        P_grid, T_grid = np.zeros((n, n)), np.zeros((n, n))

        for i, r in enumerate(rho_axis):
            for j, u in enumerate(u_axis):
                P_grid[i, j] = CP.PropsSI('P', 'D', r, 'U', u, 'Air')
                T_grid[i, j] = CP.PropsSI('T', 'D', r, 'U', u, 'Air')
        
        return (interpol.RegularGridInterpolator((rho_axis, u_axis), P_grid, bounds_error=False, fill_value=None),
                interpol.RegularGridInterpolator((rho_axis, u_axis), T_grid, bounds_error=False, fill_value=None))
    

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
        z, U_tot, T_eau, T_paroi, m_vap = Y
        
        # Strategy
        q_riv = self.f_q_riv(t)
        cmd = self.f_cmd_prix(t)
        q_pump = min(Q_p_max, q_riv) if (cmd == 1 and z < 0.9*self.H) else 0.0
        q_turb = Q_t_max if (cmd == -1 and z > 0.1*self.H) else 0.0
        q_net = q_pump - q_turb

        # Variables
        V_air = max(self.A * (self.H - z), 1e-3)
        rho = (self.m_air + m_vap) / V_air
        u = U_tot / (self.m_air + m_vap)
        P = float(self.look_P((rho, u)))
        T_air = float(self.look_T((rho, u)))

        # Heat transfers
        Q_ap, Q_ae, Q_wp, Q_ext = self._calculate_heat_transfers(z, T_air, T_eau, T_paroi)
        
        # Evaporation
        P_sat = 611.2 * np.exp(17.62 * (T_eau-273.15) / (T_eau-273.15 + 243.12))
        dm_vap_dt = self.params['k_evap'] * self.A * max(0, P_sat - (m_vap*461.5*T_air/V_air))

        # Derivatives
        dz_dt = q_net / self.A
        dW_dt = - P * (-q_net) 
        dU_tot_dt = dW_dt - Q_ae - Q_ap + (dm_vap_dt * self.params['cp_vap'] * T_eau)
        
        M_eau = max(z, 0.1) * self.A * self.params['rho_eau']
        dT_eau_dt = (Q_ae - Q_wp - (dm_vap_dt * self.params['L_v'])) / (M_eau * self.params['cp_eau'])
        dT_paroi_dt = (Q_ap + Q_wp - Q_ext) / (self.M_paroi * self.params['cp_p'])

        return [dz_dt, dU_tot_dt, dT_eau_dt, dT_paroi_dt, dm_vap_dt]

    def update(self, df_global, Q_p_max=20.0, Q_t_max=5.0, T_eau=288.15, T_paroi=293.15, m_vap=1e3):
        # Update of the global df by solving the differential equation 
        V_air0 = self.A * (self.H - self.z0)
        rho0 = self.m_air / V_air0
        
        # Initial specific energy
        u_initial_spec = CP.PropsSI('U', 'T', self.T_air0, 'D', rho0, 'Air')
        # Initial energy
        U0 = self.m_air * u_initial_spec
        #initial vector        
        Y0 = [self.z0, U0, T_eau, T_paroi, m_vap]
        
        #addition of the command functions
        self._get_inputs(df_global)
        #time in seconds
        t_span = (0, df_global['Relativ_time_h'].max() * 3600)
        t_eval = df_global['Relativ_time_h'].values * 3600
        
        sol = solve_ivp(
            fun=lambda t, Y: self._equations_differentielles(t, Y, Q_p_max, Q_t_max),
            t_span=t_span, y0=Y0, t_eval=t_eval, method='RK45', max_step=900
        )

        return self._process_results(sol, df_global)

    def _process_results(self, sol, df_global):
        # Addition of solver results
        z = sol.y[0]
        U_tot = sol.y[1]
        m_vap = sol.y[4]
        
        V_air = self.A * (self.H - z)
        rho = (self.m_air + m_vap) / V_air
        u = U_tot / (self.m_air + m_vap)
        

        df_global[f'Pression'] = self.look_P((rho, u))
        df_global[f'T_air'] = self.look_T((rho, u))
        df_global[f'z'] = z
        df_global[f'T_eau'] = sol.y[2]
        df_global[f'T_paroi'] = sol.y[3]

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