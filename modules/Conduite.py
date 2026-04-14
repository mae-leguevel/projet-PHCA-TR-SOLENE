import numpy as np
import time
import matplotlib.pyplot as plt
import fluids as fl
import CoolProp.CoolProp as CP

from base.components import BaseComponent

class Fluid:
    """Fluid Properties"""
    def __init__(self, name="Water"):
        self.name = name

    def get_properties(self, T, P):
        #Fluid properties through vectorized CoolProp
        _rho = np.vectorize(lambda t, p: CP.PropsSI('D', 'T', t, 'P', p, self.name))
        _mu = np.vectorize(lambda t, p: CP.PropsSI('V', 'T', t, 'P', p, self.name))
        _cp = np.vectorize(lambda t, p: CP.PropsSI('C', 'T', t, 'P', p, self.name))
        return _rho(T, P), _mu(T, P), _cp(T, P)

class PipelineGeometry:
    """Pipeline Geometry"""
    def __init__(self, D, L, Delta_z, roughness = 0.0035, complexity = 0, U=5):
        self.D = D  #Diameter (m)
        self.L = L  #Length (m)
        self.roughness = roughness #Roughness off the inside pipe (m)
        self.Delta_z = Delta_z  #Height between In and Out (m)
        self.complexity = complexity #number of corners
        self.U = U #Echange coeff (W.(m.K)^-1)
        self.surface = np.pi * self.D * self.L # flux surface

    
    def calculate_physics(self, Q, rho, mu):
        # Vectorized losses parameters
        v = Q / (np.pi * (self.D/2)**2)
        Re = np.array([fl.core.Reynolds(V=vel, D=self.D, mu=visc, rho=dens) for vel, visc, dens in zip(v, mu, rho)])
        eD = self.roughness / self.D
        
        # Vectorized Friction factor 
        f = np.array([fl.friction.friction_factor(re, eD) if re > 0 else 0 for re in Re])
        
        #Regular losses
        p_reg = f * (self.L / self.D) * (rho * v**2 / 2)
        
        # Ingular losses
        v_k_elbow = np.vectorize(fl.fittings.bend_rounded)
        K_sin = v_k_elbow(Di=self.D, angle=90, rc=1.5*self.D, Re=Re, fd=f)
        p_sin = self.complexity * K_sin * (rho * v**2 / 2)
        
        p_alt = rho * 9.81 * self.Delta_z
        return p_reg + p_sin + p_alt


class Conduit(BaseComponent):
    def __init__(self, name, D, L, Delta_z, start, stop, fluid_name="Water", roughness=0.0035, complexity=2):
        super().__init__(name=name, start=start, stop=stop)
        #Composition
        self.geometry = PipelineGeometry(D, L, Delta_z, roughness, complexity)
        self.fluid = Fluid(fluid_name)

    def fetch_data(self):
        """
        Nothing here
        """
        pass

    def update(self, df, q_name):
        """
        Df Thermal and Mecanical update
        """
        Q = df.get(q_name, np.zeros(len(df))).values
        T_in = np.full(len(df), 283.15)
        P_in = np.full(len(df), 1.01325e5) # Atm pressure by default
        T_ext = np.full(len(df), 288.15) # External (underground) temperature

        # 1. Fluid Properties
        rho, mu, Cp = self.fluid.get_properties(T_in, P_in)

        # 2. Pressure drop
        p_losses = self.geometry.calculate_physics(Q, rho, mu)

        # 3. Thermal exange
        phi_total = (p_losses * Q) + (self.geometry.U * self.geometry.surface * (T_ext - T_in))
        
        #Temperature variation
        dT = np.where(Q > 0, phi_total / (rho * Q * Cp), 0)
        T_out = T_in + dT

        #Saving in the df
        df[f"{self.name}_DeltaP"] = p_losses
        df[f"{self.name}_T_out"] = T_out
        
        return df