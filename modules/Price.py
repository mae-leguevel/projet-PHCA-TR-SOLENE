import pandas as pd
from base.components import BaseComponent

class Price(BaseComponent):
    def __init__(self, path, start="2025-01-01", stop="2025-01-31"):
        super().__init__(name="Price", start=start, stop=stop)
        self.path = path
        self.types_demandes = ['Price (EUR/MWhe)']

    def fetch_data(self):
        df = pd.read_csv(self.path)[['Price (EUR/MWhe)', 'Datetime (Local)']]
        df['Date'] = pd.to_datetime(df['Datetime (Local)'])
        
        mask = (df['Date'] >= self.start) & (df['Date'] < self.stop)
        self.df_source = df.loc[mask].sort_values('Date')

        t0 = pd.to_datetime(self.start)
        self.df_source['t_relative'] = (self.df_source['Date'] - t0).dt.total_seconds() / 3600.0
        
        print(f"Données de prix chargées : {len(self.df_source)} points.")

    def compute_thresholds(self, morning=("07:00", "10:59"), evening=("17:00", "22:59")):
        if self.df_source is None:
            raise ValueError("Lancez fetch_data() avant compute_thresholds().")

        temp_df = self.df_source.set_index('Date')
        
        daily_mean = temp_df['Price (EUR/MWhe)'].resample('D').mean()

        peak_m = temp_df.between_time(*morning)
        peak_e = temp_df.between_time(*evening)
        peak_avg = pd.concat([peak_m, peak_e])['Price (EUR/MWhe)'].resample('D').mean()

        day_keys = self.df_source['Date'].dt.normalize()
        self.df_source['Mean_Daily'] = day_keys.map(daily_mean)
        self.df_source['Peak_Daily'] = day_keys.map(peak_avg)

    def apply_strategy(self, method='threshold'):
        self.df_source['Price_command'] = 0
        
        if method == 'threshold':
            low_price_mask = self.df_source['Price (EUR/MWhe)'] < self.df_source['Mean_Daily']
            self.df_source.loc[low_price_mask, 'Price_command'] = 1
            
            high_price_mask = self.df_source['Price (EUR/MWhe)'] > self.df_source['Peak_Daily']
            self.df_source.loc[high_price_mask, 'Price_command'] = -1

        elif method == 'simplified':
            hours = self.df_source['Date'].dt.hour
            
            turb_mask = ((hours >= 7) & (hours < 11)) | ((hours >= 17) & (hours < 23))
            self.df_source.loc[turb_mask, 'Price_command'] = -1
            

            pump_mask = (hours >= 11) & (hours < 17)
            self.df_source.loc[pump_mask, 'Price_command'] = 1
            
        self.types_demandes += ['Price_command', 'Mean_Daily', 'Peak_Daily']
        print(f"Stratégie de prix '{method}' appliquée.")

    def update(self, df_global):
        if not self.interpolators:
            self.create_interpolators(self.types_demandes)
        return self.update_global_df(df_global)