import mysql.connector

conn = mysql.connector.connect(host='localhost', user='root', password='Bant@6963', database='lon-default')
cur = conn.cursor()
cur.execute("SELECT THREAD_ID, OBJECT_TYPE, OBJECT_SCHEMA, OBJECT_NAME, LOCK_TYPE, LOCK_STATUS FROM performance_schema.metadata_locks WHERE OBJECT_SCHEMA='lon-default' AND OBJECT_NAME='cases'")
print('METADATA LOCKS:')
for row in cur.fetchall():
    print(row)
cur.execute('SHOW FULL PROCESSLIST')
print('\nPROCESSLIST:')
for row in cur.fetchall():
    print(row)
conn.close()
