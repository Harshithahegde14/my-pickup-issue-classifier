import sqlite3

# ----- Create predictions.db -----
conn1 = sqlite3.connect('predictions.db')
cursor1 = conn1.cursor()

cursor1.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    department TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn1.commit()
conn1.close()

# ----- Create my_pickup.db -----
conn2 = sqlite3.connect('my_pickup.db')
cursor2 = conn2.cursor()

cursor2.execute('''
CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    department TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn2.commit()
conn2.close()

print("✅ Both databases and tables have been created successfully!")
