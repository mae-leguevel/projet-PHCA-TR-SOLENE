# base/component.py
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
import pandas as pd

class BaseComponent(ABC):
    def __init__(self, name, start, stop):
        self.name = name
        self.start = start
        self.stop = stop
        self.df_source = None  
        self.interpolators = {}

    @abstractmethod
    def fetch_data(self, *args, **kwargs):
        """Récupère les données (API, CSV, ou Géo)"""
        pass

    def create_interpolators(self, types, t_col='t_relative'):
        """Génère le dictionnaire de fonctions à partir de df_source"""
        if self.df_source is None:
            raise ValueError(f"Aucune donnée source pour {self.name}. Lancez fetch_data() d'abord.")
            
        time = self.df_source[t_col].values
        for t in types:
            values = self.df_source[t].values
            self.interpolators[t] = interp1d(
                time, values, kind='previous', 
                bounds_error=False, fill_value=(values[0], values[-1])
            )
        return self.interpolators

    def update_global_df(self, df_global):
        """Remplit le DataFrame principal avec les fonctions d'interpolation"""
        for t, func in self.interpolators.items():
            df_global[f"{self.name}_{t}"] = func(df_global['Temps relatif (h)'])
        return df_global