import mysql.connector
conn = mysql.connector.connect(host='localhost', user='root', password='Bant@6963', database='lon-default')
cur = conn.cursor()
cur.execute('SHOW COLUMNS FROM performance_schema.metadata_locks')
for row in cur.fetchall():
    print(row)
conn.close()
