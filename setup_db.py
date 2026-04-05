import sqlite3

conn = sqlite3.connect('inventory.db')  # Persist to file so all agents can access it
cursor = conn.cursor()

cursor.execute('CREATE TABLE IF NOT EXISTS inventory (item TEXT PRIMARY KEY, stock INTEGER)')
# cursor.execute("""
#     INSERT INTO inventory VALUES
#     ('WidgetA', 15),
#     ('WidgetB', 10),
#     ('GadgetX', 5),
#     ('FakeItem', 0)
# """)

cursor.execute('SELECT * from inventory')
all_rows = cursor.fetchall()
print(all_rows)

conn.commit()

