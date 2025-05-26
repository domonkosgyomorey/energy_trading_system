class Config:
    DB_PATH = "household_dbs"
    DB = "merged2024.parquet"
    FORECASTER_HIST_SIZE = 10
    FORECASTER_PRED_SIZE = 10
    HOUSEHOLD_WITHOUT_BATTERY_PROB = 0.8
    CITY_GRID_PRICE_PRED_SIZE = 10
