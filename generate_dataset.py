import pandas as pd

# Sample synthetic data
df = pd.DataFrame({
    'transaction_id': range(1, 13),
    'amount': [4500, 250000, 3800, 175000, 2000, 500000, 30000, 7500, 120000, 1800, 95000, 400000],
    'location': ['Mumbai', 'Hyderabad', 'Bangalore', 'Dubai', 'Chennai', 'Singapore', 'Delhi', 'Hyderabad', 'London', 'Kolkata', 'Mumbai', 'Panama'],
    'timestamp': ['2024-11-20 14:35:00', '2024-11-21 02:15:00', '2024-11-22 10:05:00', '2024-11-23 01:10:00',
                  '2024-11-23 12:45:00', '2024-11-23 03:25:00', '2024-11-23 09:30:00', '2024-11-24 15:00:00',
                  '2024-11-24 01:50:00', '2024-11-25 14:20:00', '2024-11-25 23:10:00', '2024-11-26 00:05:00'],
    'business_account': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    'label': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1]
})

# Feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['location_flag'] = df['location'].apply(lambda x: 1 if x in ['Dubai', 'Singapore', 'London', 'Panama'] else 0)
df['odd_time'] = df['timestamp'].dt.hour.apply(lambda x: 1 if 0 <= x <= 4 else 0)
df['large_amount'] = df['amount'].apply(lambda x: 1 if x > 100000 else 0)

# Save the processed dataset
df.to_csv('processed_transactions.csv', index=False)
print("✅ processed_transactions.csv created successfully.")
