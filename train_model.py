import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("1.csv")

# === только выполненные заказы ===
if 'Booking Status' in df.columns:
    df = df[df['Booking Status'].str.lower() == 'completed']

# === исключение аномалий ===
df = df[(df['Ride Distance'] > 0) & (df['Ride Distance'] < 300)]
df = df[(df['Booking Value'] > 0) & (df['Booking Value'] < 2000)]
df = df[~((df['Ride Distance'] > 30) & (df['Booking Value'] < 300))]
df = df[df['Booking Value'] / df['Ride Distance'] >= 5]

# === оптимальный тариф через линейную регрессию ===
X_tariff = df[['Ride Distance']]
y_tariff = df['Booking Value']

lin_reg = LinearRegression()
lin_reg.fit(X_tariff, y_tariff)

BASE_FARE = lin_reg.intercept_
FARE_PER_KM = lin_reg.coef_[0]

print(f"📊 Оптимальный тариф из данных: BASE_FARE={BASE_FARE:.2f}, FARE_PER_KM={FARE_PER_KM:.2f}")

# === expected_fare и delta ===
df['expected_fare'] = BASE_FARE + df['Ride Distance'] * FARE_PER_KM
df['delta'] = df['Booking Value'] - df['expected_fare']

# === балансировка данных ===
df = df[df['Booking Value'] >= 0.5 * df['expected_fare']]
df['sample_weight'] = np.where(df['Booking Value'] > df['expected_fare'], 2.0, 1.0)


df['pickup_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['month'] = df['pickup_datetime'].dt.month

def time_of_day(hour):
    if 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 18: return 'day'
    elif 18 <= hour < 24: return 'evening'
    else: return 'night'

df['time_of_day'] = df['pickup_hour'].apply(time_of_day)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_peak_hour'] = df['pickup_hour'].isin([7,8,9,17,18,19,20]).astype(int)

df['traffic_level'] = np.random.randint(1, 6, size=len(df))
df['distance_traffic'] = df['Ride Distance'] * df['traffic_level']
df['hour_weekend'] = df['pickup_hour'] * df['is_weekend']
df['value_per_km'] = df['Booking Value'] / df['Ride Distance']
df['avg_pickup_value'] = df.groupby('Pickup Location')['Booking Value'].transform('mean')
df['avg_drop_value'] = df.groupby('Drop Location')['Booking Value'].transform('mean')

for col in ['Pickup Location', 'Drop Location']:
    top_values = df[col].value_counts().nlargest(30).index
    df[col] = df[col].where(df[col].isin(top_values), 'Other')

features = ['Ride Distance', 'pickup_hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
            'traffic_level', 'Pickup Location', 'Drop Location',
            'Payment Method', 'Driver Ratings', 'Customer Rating',
            'distance_traffic', 'hour_weekend', 'value_per_km',
            'avg_pickup_value', 'avg_drop_value', 'time_of_day']

categorical_features = ['Pickup Location', 'Drop Location', 'Payment Method', 'time_of_day']
numeric_features = ['Ride Distance', 'pickup_hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                    'traffic_level', 'Driver Ratings', 'Customer Rating',
                    'distance_traffic', 'hour_weekend', 'value_per_km',
                    'avg_pickup_value', 'avg_drop_value']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# === обучение randomforest для каждого типа транспорта ===
vehicle_types = df['Vehicle Type'].unique()

for vtype in vehicle_types:
    df_sub = df[df['Vehicle Type'] == vtype]
    if df_sub.empty:
        continue

    X = df_sub[features]
    y = df_sub['delta']
    weights = df_sub['sample_weight']

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train, regressor__sample_weight=w_train)

    y_pred_delta = pipeline.predict(X_test)
    y_pred_final = y_pred_delta + df_sub.loc[y_test.index, 'expected_fare']

    print(f"\n=== {vtype} ===")
    print("MAE:", mean_absolute_error(df_sub.loc[y_test.index, 'Booking Value'], y_pred_final))
    print("R²:", r2_score(df_sub.loc[y_test.index, 'Booking Value'], y_pred_final))

    joblib.dump(pipeline, f"models/randomforest_delta_dynamic_model_{vtype.lower()}.pkl")
    print(f"{vtype} сохранена в models/randomforest_delta_dynamic_model_{vtype.lower()}.pkl")
