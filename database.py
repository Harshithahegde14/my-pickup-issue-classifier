import sqlite3

# Connect to SQLite database (creates if it doesn't exist)
conn = sqlite3.connect('my_pickup.db')
cursor = conn.cursor()

# Create a table to store incoming messages and predictions
cursor.execute('''
    CREATE TABLE IF NOT EXISTS customer_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        department TEXT NOT NULL,
        category TEXT NOT NULL
    )
''')

# Save changes and close connection
conn.commit()
conn.close()

print("Database and table created successfully!")

