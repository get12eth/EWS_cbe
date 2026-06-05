import mysql.connector

conn = mysql.connector.connect(host='localhost', user='root', password='Bant@6963', database='lon-default')
cur = conn.cursor()
cur.execute('SHOW FULL PROCESSLIST')
print('PROCESSLIST:')
for row in cur.fetchall():
    print(row)
print('\nLOCKS:')
cur.execute('SELECT * FROM information_schema.INNODB_LOCKS LIMIT 20')
for row in cur.fetchall():
    print(row)
print('\nWAITS:')
cur.execute('SELECT * FROM information_schema.INNODB_LOCK_WAITS LIMIT 20')
for row in cur.fetchall():
    print(row)
conn.close()
