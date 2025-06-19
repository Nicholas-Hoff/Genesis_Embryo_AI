import duckdb
import os
import logging
from logging_config import configure_logging

logger = logging.getLogger(__name__)

# 1) Point at source files
src_files = [
    'godseed_aggressive_shadow.db',
    'godseed_aggressive_full.db',
 #   'godseed_aggressive_full.db',
]
target_db = 'godseed_training.db'

# 2) Make sure we start fresh
if os.path.exists(target_db):
    os.remove(target_db)

# 3) Discover table names by opening the first DB on its own
configure_logging()
logger.info(f"Discovering tables in {src_files[0]}")
con0 = duckdb.connect(src_files[0])
tables = [row[0] for row in con0.execute("SHOW TABLES;").fetchall()]
con0.close()

if not tables:
    raise RuntimeError(f"No tables found in {src_files[0]} – are you sure it has tables?")
logger.info(f"Found tables: {tables}")

# 4) Now open (and create) the target DB and attach all three sources
con = duckdb.connect(target_db)
for idx, fn in enumerate(src_files):
    schema = f"src{idx}"
    con.execute(f"ATTACH '{fn}' AS {schema};")
    logger.info(f"Attached {fn} -> schema {schema}")

# 5) For each table, UNION ALL across src0, src1, src2
for t in tables:
    logger.info(f"Merging table `{t}`…")
    union_sql = " UNION ALL ".join(f"SELECT * FROM src{idx}.{t}" for idx in range(len(src_files)))
    con.execute(f"CREATE TABLE {t} AS {union_sql};")

# 6) Optional cleanup
con.execute("VACUUM;")
con.close()

logger.info(f"Done! Merged {len(tables)} tables into `{target_db}`")
