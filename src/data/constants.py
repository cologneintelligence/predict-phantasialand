"""
Project: Phantasialand  
State: 10/2021

Various constants that are needed by data scripts.
"""

from pathlib import Path

DATA_PATH = (Path(__file__).parent.parent.parent / "data").resolve()

LOGGING_FORMAT_STR = "[%(asctime)s] %(message)s"

# mapping names of Phantasialand attractions to the internal ids used by wartezeiten.app
WARTEZEITEN_APP_ATTRACTIONS = {
    "Taron": "636a733d",
    "Black Mamba": "646a383d",
    "Colorado Adventure": "6354673d",
    "Feng Ju Palace": "647a773d",
    "Maus-au-Chocolat": "64773d3d",
    "Mystery Castle": "636a553d",
    "Bolles Riesenrad": "63513d3d",
    "Bumper Klumpen": "6444303d",
    "Chiapas DIE Wasserbahn": "6354553d",
    "Crazy Bats": "6444343d",
    "Der lustige Papagei": "6454343d",
    "Die fröhliche Bienchenjagd": "64546b3d",
    "Geister Rikscha": "647a383d",
    "Pferdekarussell": "63413d3d",
    "Raik": "636a6f3d",
    "River Quest": "636a513d",
    "Talocan": "6354733d",
    "Tikal": "63546f3d",
    "Verrücktes Hotel Tartüff": "64513d3d",
    "Wellenflug": "64673d3d",
    "Winja´s Fear": "646a513d",
    "Winja´s Force": "6454303d",
    "Wirtl´s Taubenturm": "6454553d",
    "Wolke´s Luftpost": "6454673d",
    "Wupi´s Wabi Wipper": "6444383d",
    "Wözl´s Duck Washer": "6444773d",
    "Würmling Express": "6454383d",
    "Tittle Tattle Tree": "6454773d",
    "F.L.Y.": "63543038",
    "Bolles Flugschule": "63673d3d",
    "Hollywood Tour": "64446b3d",
    "Wakobato": "6444673d",
    "Wözl´s Wassertreter": "6454513d",    
}

# mapping months of the year to the internal ids used by wartezeiten.app
WARTEZEITEN_APP_MONTHS = {
    "January": "63673d3d",
    "May": "64673d3d",
    "June": "64513d3d",
    "July": "64413d3d",
    "August": "65773d3d",
    "September": "65673d3d",
    "October": "636a303d",
    "November": "636a773d",
    "December": "636a383d",
}

# mapping years to the internal ids used by wartezeiten.app
WARTEZEITEN_APP_YEARS = {
    "2021": "6354303965413d3d",
    "2020": "6354303965513d3d",
    "2019": "6354302b63413d3d",
}

# mapping names of federal states to their ISO code
STATE_FULL2ISO = {
    "Baden-Württemberg": "BW",
    "Bayern": "BY",
    "Berlin": "BE",
    "Brandenburg": "BB",
    "Bremen": "HB",
    "Hamburg": "HH",
    "Hessen": "HE",
    "Mecklenburg-Vorpommern": "MV",
    "Niedersachsen": "NI",
    "Nordrhein-Westfalen": "NW",
    "Rheinland-Pfalz": "RP",
    "Saarland": "SL",
    "Sachsen": "SN",
    "Sachsen-Anhalt": "ST",
    "Schleswig-Holstein": "SH",
    "Thüringen": "TH",
}

# mapping the original column names of the Deutscher Wetterdienst to more descriptive
# names
DWD_COLUMN_NAMES2DESCRIPTION = {
    "STATIONS_ID": "station_id",
    "MESS_DATUM": "date",
    "QN_3": "quality_wind",
    "FX": "max_wind_gust",
    "FM": "mean_wind_speed",
    "QN_4": "quality_other",
    "RSK": "precipitation_height",
    "RSKF": "precipitation_form",
    "SDK": "sunshine_duration",
    "SHK_TAG": "snow_depth",
    "NM": "mean_cloud_cover",
    "VPM": "mean_vapor_pressure",
    "PM": "mean_pressure",
    "TMK": "mean_temperature",
    "UPM": "mean_relative_humidity",
    "TXK": "max_temperature_2m",
    "TNK": "min_temperature_2m",
    "TGK": "min_temperature_5cm",
}
