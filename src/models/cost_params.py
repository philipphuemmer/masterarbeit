"""
Gemeinsame Kostenparameter für alle Planungsmodelle (Myopic, CFA, VFA).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostParams:
    """
    Wirtschaftliche Kostenparameter für die Wartungsplanung.

    Gelten für alle Modelle gleichermaßen.

    Parameters
    ----------
    wage_eur_per_hour : float
        Stundenlohn pro Team [€/h]. Standard: 40 €/h.
    fuel_eur_per_km : float
        Kraftstoffkosten [€/km]. Standard: 0,30 €/km.
    downtime_eur_per_kwh : float
        Ausfallkosten pro kWh nicht gelieferter Energie [€/kWh].
        Wird multipliziert mit Nennleistung [kW] × Wartezeit [h].
        Standard: 0,50 €/kWh.
    typ1_service_min : float
        Servicezeit Typ-1-Störung [min]. Standard: 60 min.
    typ2_dismount_min : float
        Demontagezeit Typ-2-Störung [min]. Standard: 30 min.
    typ2_handling_min : float
        Lagerhandling am Depot für Typ-2-Störung [min]. Standard: 5 min.
    typ2_remount_min : float
        Montagezeit Typ-2-Störung [min]. Standard: 30 min.
    """

    wage_eur_per_hour: float = 40.0
    fuel_eur_per_km: float = 0.30
    downtime_eur_per_kwh: float = 0.50
    typ1_service_min: float = 60.0
    typ2_dismount_min: float = 30.0
    typ2_handling_min: float = 5.0
    typ2_remount_min: float = 30.0
