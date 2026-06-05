from pathlib import Path
import re
import mysql.connector

sql_path = Path('sql/comprehensive_schema.sql')
if not sql_path.exists():
    raise SystemExit(f'SQL file not found: {sql_path}')

sql = sql_path.read_text(encoding='utf-8')
cleaned_lines = []
for line in sql.splitlines():
    stripped = line.strip()
    if stripped.startswith('--'):
        continue
    cleaned_lines.append(re.sub(r'--.*$', '', line))
cleaned = '\n'.join(cleaned_lines)
statements = [stmt.strip() for stmt in cleaned.split(';') if stmt.strip()]
print('statement_count', len(statements))

conn = mysql.connector.connect(host='localhost', user='root', password='Bant@6963', database='lon-default')
cur = conn.cursor()
for idx, stmt in enumerate(statements, 1):
    if idx <= 5 or idx % 10 == 0:
        print('executing', idx, stmt.splitlines()[0][:120])
    cur.execute(stmt)
conn.commit()
conn.close()
print('done', len(statements))
