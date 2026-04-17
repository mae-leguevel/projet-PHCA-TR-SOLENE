import requests
import pandas as pd
from base.components import BaseComponent

class River(BaseComponent):
    def __init__(self, code, name, start="2025-01-01", stop="2025-01-31"):
        super().__init__(name=name, start=start, stop=stop)
        self.code_station = code
        self.types_demandes = []

    def fetch_data(self, q_types=["QmnJ", "QINnJ"], include_temp=True):
        """Récupère Q et T, et prépare le df_source unique"""
        self.types_demandes = q_types
        
        
        df_q = self._load_hydrometry(q_types)
        
        df_t = None
        if include_temp:
            df_t = self._load_temperature()
            self.types_demandes.append("Temp_Eau")

        if df_q is not None and df_t is not None:
            df_q = df_q.sort_values('Date')
            df_t = df_t.sort_values('Date')
            self.df_source = pd.merge_asof(df_q, df_t, on='Date', direction='nearest')
        elif df_q is not None:
            self.df_source = df_q
        
       
        t0 = pd.to_datetime(self.start)
        diff = self.df_source['Date'] - t0
        self.df_source['t_relative'] = diff.dt.total_seconds() / 3600.0
        
        print(f"Initialisation de {self.name} terminée.")

    def _load_hydrometry(self, types):
        url_api = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
        main_df = None
        for t in types:
            params = {
                "code_entite": self.code_station,
                "grandeur_hydro_elab": t, 
                "date_debut_obs_elab": self.start, 
                "date_fin_obs_elab": self.stop,                     
            }
            resp = requests.get(url_api, params=params)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if data:
                    temp_df = pd.DataFrame(data)[['date_obs_elab', 'resultat_obs_elab']]
                    temp_df['resultat_obs_elab'] = temp_df['resultat_obs_elab'] / 1000.0 # L.s^-1 -> m^3.s^-1
                    temp_df.columns = ['Date', t]
                    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                    main_df = temp_df if main_df is None else pd.merge(main_df, temp_df, on='Date', how='outer')
        return main_df

    def _load_temperature(self):
        url_api = "https://hubeau.eaufrance.fr/api/v1/temperature/chronique"
        params = {"code_station": self.code_station, "date_debut_mesure": self.start, "date_fin_mesure": self.stop, "size": 5000}
        resp = requests.get(url_api, params=params)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            if data:
                df = pd.DataFrame(data)
                df['Date'] = pd.to_datetime(df['date_mesure_temp'] + ' ' + df['heure_mesure_temp'])
                df = df[['Date', 'resultat']].rename(columns={'resultat': 'Temp_Eau'})
                return df
        return None

    def update(self, df_global):
        if not self.interpolators:
            self.create_interpolators(self.types_demandes)
        return self.update_global_df(df_global)
    
    def apply_strategy(self, source_col='QmnJ', method='percentage', param=0.1, Qin=5, Qout=5):
        if self.df_source is None:
            raise ValueError("df_source est vide. Lancez fetch_data() avant d'appliquer une stratégie.")

        if method == 'percentage':
            self.df_source['Q_prelevable'] = self.df_source[source_col] * param
            
        elif method == 'threshold':
            self.df_source['Q_prelevable'] = (self.df_source[source_col] - param).clip(lower=0)
        
        elif method == "simplified":
            self.df_source['Q_prelevable'] = param
        
        else:
            raise ValueError(f"Méthode {method} non reconnue.")
        if 'Q_prelevable' not in self.types_demandes:
            self.types_demandes.append('Q_prelevable')
            
        print(f"Stratégie '{method}' appliquée sur {source_col}. Colonne 'Q_prelevable' créée.")
