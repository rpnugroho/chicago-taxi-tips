# Store configuration here

# Feature
NUMERIC_FEATURES = [
    'trip_miles', 'fare', 'trip_seconds'
]


ORDINAL_FEATURES = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month'
]

CATEGORICAL_FEATURES = [
    'company', 'payment_type', 'pickup_community_area',
    'dropoff_community_area'
]


BIN_FEATURES = [
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude'
]


DROP_FEATURES = [
    'pickup_census_tract', 'dropoff_census_tract'
]


TARGET = 'target'
