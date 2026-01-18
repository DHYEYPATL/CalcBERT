# checkDB.py
import sqlite3
import json
from tabulate import tabulate  # pip install tabulate

DB_PATH = "backend/backend_feedback.db"  # Update if your DB is elsewhere

def main():
    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Found tables: {tables}\n")

    for table in tables:
        print(f"=== Table: {table} ===")

        # Get columns
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [col[1] for col in cursor.fetchall()]

        # Fetch all rows
        cursor.execute(f"SELECT * FROM {table};")
        rows = cursor.fetchall()

        # Process rows: pretty print JSON columns
        processed_rows = []
        for row in rows:
            row_list = list(row)
            for i, val in enumerate(row_list):
                if isinstance(val, str) and val.startswith("{") and val.endswith("}"):
                    try:
                        row_list[i] = json.dumps(json.loads(val), indent=2, ensure_ascii=False)
                    except Exception:
                        pass  # Leave as is if not valid JSON
            processed_rows.append(row_list)

        if processed_rows:
            # Identify JSON columns
            json_cols = [i for i, col in enumerate(cols) if 'json' in col.lower()]

            # Print table with JSON replaced by placeholder
            table_rows = [
                [val if i not in json_cols else '<<JSON>>' for i, val in enumerate(row)]
                for row in processed_rows
            ]
            print(tabulate(table_rows, headers=cols, tablefmt="grid"))

            # Print full JSON columns separately
            for row in processed_rows:
                for i in json_cols:
                    print(f"\n--- {cols[i]} ---\n{row[i]}\n")

        else:
            print("No data in this table.")

        print("\n")  # Spacer between tables

    conn.close()


if __name__ == "__main__":
    main()
