import joblib
import pandas as pd
from datetime import datetime
import warnings
import os
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=UserWarning)


available_models = [f for f in os.listdir("models") if f.startswith("randomforest_delta_dynamic_model_") and f.endswith(".pkl")]
vehicle_types = [m.replace("randomforest_delta_dynamic_model_", "").replace(".pkl", "") for m in available_models]

if not vehicle_types:
    print("❌ Нет обученных RandomForest моделей в папке models/")
    exit()


print("\nДоступные типы машин:")
for i, vtype in enumerate(vehicle_types, start=1):
    print(f"{i}. {vtype}")

try:
    choice = int(input("\nВведите номер типа машины: "))
    if choice < 1 or choice > len(vehicle_types):
        raise ValueError
    vehicle_type = vehicle_types[choice - 1]
except ValueError:
    print("❌ Неверный выбор.")
    exit()


model_path = f"models/randomforest_delta_dynamic_model_{vehicle_type}.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"❌ Модель для {vehicle_type} не найдена.")
    exit()

# === загрузка датасета для статистики и тарифа (чтобы сравнивать в итоге) ===
try:
    df = pd.read_csv("1.csv")
    df = df.dropna(subset=['Ride Distance', 'Booking Value'])
except FileNotFoundError:
    print("❌ Датасет 1.csv не найден.")
    df = None


try:
    distance_km = float(input("Введите дистанцию поездки (км): "))
except ValueError:
    print("❌ Неверный формат дистанции.")
    exit()

# === генерация признаков ===
now = datetime.now()
hour = now.hour
day_of_week = now.weekday()
month = now.month

is_peak = 1 if hour in [7,8,9,17,18,19,20] else 0
is_weekend = 1 if day_of_week >= 5 else 0
traffic_level = 3
distance_traffic = distance_km * traffic_level
hour_weekend = hour * is_weekend
value_per_km = 0

pickup_location = "CityCenter"
drop_location = "Airport"
payment_method = "Card"
driver_rating = 4.5
customer_rating = 4.7
avg_pickup_value = 400
avg_drop_value = 420

time_of_day = (
    "morning" if 6 <= hour < 12 else
    "day" if 12 <= hour < 18 else
    "evening" if 18 <= hour < 24 else
    "night"
)


if df is not None and not df.empty:
    lin_reg = LinearRegression()
    lin_reg.fit(df[['Ride Distance']], df['Booking Value'])
    BASE_FARE = lin_reg.intercept_
    FARE_PER_KM = lin_reg.coef_[0]
else:
    BASE_FARE = 30
    FARE_PER_KM = 8

expected_fare = BASE_FARE + distance_km * FARE_PER_KM

def make_input():
    return pd.DataFrame([{
        'Ride Distance': distance_km,
        'pickup_hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak,
        'traffic_level': traffic_level,
        'Pickup Location': pickup_location,
        'Drop Location': drop_location,
        'Vehicle Type': vehicle_type,
        'Payment Method': payment_method,
        'Driver Ratings': driver_rating,
        'Customer Rating': customer_rating,
        'distance_traffic': distance_traffic,
        'hour_weekend': hour_weekend,
        'value_per_km': value_per_km,
        'avg_pickup_value': avg_pickup_value,
        'avg_drop_value': avg_drop_value,
        'time_of_day': time_of_day
    }])

# === Прогноз финальной цены ===
predicted_price = model.predict(make_input())[0]
price_per_km = predicted_price / distance_km if distance_km > 0 else 0

print(f"\n💡 Прогноз для {vehicle_type}:")
print(f"Предсказанная моделью цена ≈ {abs(predicted_price):.2f} руб")
print(f"Цена за км ≈ {abs(price_per_km):.2f} руб/км")
print(f"Тарифная цена (expected_fare) ≈ {expected_fare:.2f} руб")

# === Сравнение с данными ===
if df is not None and not df.empty:
    df_filtered = df[(df['Vehicle Type'].str.lower() == vehicle_type.lower()) &
                     (df['Ride Distance'] >= distance_km - 1) &
                     (df['Ride Distance'] <= distance_km + 1)]
    if df_filtered.shape[0] > 0:
        mean_price = df_filtered['Booking Value'].mean()
        median_price = df_filtered['Booking Value'].median()
        print(f"\n📊 Статистика по {vehicle_type} на ~{distance_km:.0f} км:")
        print(f"Средняя цена ≈ {mean_price:.2f} руб")
        print(f"Медианная цена ≈ {median_price:.2f} руб")
    else:
        print("\n📊 В датасете нет поездок с такой дистанцией для сравнения.")

if distance_km > 100:
    print("⚠️ Внимание: дистанция выходит за диапазон тренировочных данных.")
