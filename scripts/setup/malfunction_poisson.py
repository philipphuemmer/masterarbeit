import numpy as np
import pandas as pd

output_path = "data/malfunction.csv"

def simulate_charging_stations(days, num_stations=20):
    # Definierte Arbeitszeit: 8:00 bis 16:00 (9 Stunden-Slots)
    operating_hours = list(range(8, 17)) 
    hours_per_day = len(operating_hours)
    
    # Raten pro Stunde (3/9 und 1/9)
    lambda_1 = 3 / hours_per_day
    lambda_2 = 1 / hours_per_day
    
    stations = [f"{i:02d}" for i in range(1, num_stations + 1)]
    all_events = []
    
    for day in range(1, days + 1):
        # Störungen für alle 9 Stunden-Slots gleichzeitig würfeln
        type_1_counts = np.random.poisson(lambda_1, hours_per_day)
        type_2_counts = np.random.poisson(lambda_2, hours_per_day)
        
        for idx, hour in enumerate(operating_hours):
            # Typ 1 Störungen dieser Stunde
            for _ in range(type_1_counts[idx]):
                all_events.append({
                    "Tag": day,
                    "Uhrzeit": hour,
                    "Typ": "Typ 1",
                    "Station_ID": np.random.choice(stations)
                })
                
            # Typ 2 Störungen dieser Stunde
            for _ in range(type_2_counts[idx]):
                all_events.append({
                    "Tag": day,
                    "Uhrzeit": hour,
                    "Typ": "Typ 2",
                    "Station_ID": np.random.choice(stations)
                })
    
    return pd.DataFrame(all_events)

# Simulation für 5 Tage zum Testen
df_maintenance = simulate_charging_stations(days=100, num_stations=397)

df_maintenance.to_csv(output_path, index=False, encoding="utf-8-sig")

# Ausgabe sortiert nach Tag und Uhrzeit
print(df_maintenance.sort_values(by=["Tag", "Uhrzeit"]))