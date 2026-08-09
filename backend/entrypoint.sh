#!/bin/sh
set -e

python <<'PYEOF'
import os
import sys
import time

import psycopg

url = os.environ["DATABASE_URL"]
for attempt in range(30):
    try:
        psycopg.connect(url).close()
        break
    except psycopg.OperationalError:
        print("Waiting for database...", file=sys.stderr)
        time.sleep(1)
else:
    sys.exit("Database not available after 30s")
PYEOF

python manage.py migrate --noinput

exec python manage.py runserver 0.0.0.0:8000
