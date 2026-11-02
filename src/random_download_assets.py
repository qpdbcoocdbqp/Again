from src.catalog import connect
from src.simready import download_asset

cli = connect()
cur = cli.execute(
"SELECT * FROM assets ORDER BY random() LIMIT 3"
)
cols, data = [d[0] for d in cur.description], cur.fetchall()

for row in data:
    print(f"{cols[0]}: {row[0]}")

print("Downloading:", row[0])
download_asset(asset=row[0], textures=True)

cur.close()
cli.close()
