import pandas as pd
from vanna.base import VannaBase
from vanna.chromadb import ChromaVectorStore
import sqlite3
import os
import shutil

# 1. Setup and Define a Vanna Class
#    -  We'll create a class that inherits from VannaBase and ChromaVectorStore
#    -  This sets up Vanna to use Chroma for storing training data.


class MyVanna(VannaBase, ChromaVectorStore):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        ChromaVectorStore.__init__(self, config=config)


# 2. Initialize Vanna
#    -  Instantiate the Vanna class we defined.
#    -  Optionally, specify the path for ChromaDB storage.  If the path exists, delete it.
persist_dir = "chroma_db"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)  # Delete if exists
vn = MyVanna(config={"persist_directory": persist_dir})

# 3. Connect to a Database
#    -  Here, we'll use an in-memory SQLite database for simplicity.
#    -  Vanna can connect to various databases (Postgres, etc.).
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# 4. Create a Table
#    -  Define a simple table schema.
cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name VARCHAR(255),
        department VARCHAR(255),
        salary INTEGER
    )
""")

# 5. Insert Data
#    -  Populate the table with some sample data.
data = [
    (1, 'Alice', 'Sales', 50000),
    (2, 'Bob', 'Sales', 60000),
    (3, 'Charlie', 'Engineering', 70000),
    (4, 'David', 'Engineering', 80000),
    (5, 'Eve', 'Marketing', 55000),
    (6, 'Frank', 'Marketing', 65000),
    (7, 'Grace', 'HR', 60000),
    (8, 'Henry', 'HR', 70000),
]
cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", data)
conn.commit()

# 6. Train Vanna
#    -  Provide Vanna with information about the database.
#    -  This is crucial for Vanna to generate accurate SQL.
#    -  We'll train it with:
#        -   DDL (Data Definition Language): The table schema.
#        -   Sample SQL queries: Examples of how to query the data.
#        -    Documentation
vn.train(ddl="""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name VARCHAR(255),
        department VARCHAR(255),
        salary INTEGER
    )
""")

vn.train(sql="SELECT department, AVG(salary) FROM employees GROUP BY department")
vn.train(sql="SELECT name FROM employees WHERE salary > 60000")

vn.train(documentation="""
    The employees table contains information about employees,
    including their id, name, department, and salary.
""")

# 7. (Optional) Add a visualization.
#    - Vanna can also store information about how to visualize the data.
#    -  This is optional, but can be helpful.
chart_sql = """
SELECT
    department,
    AVG(salary) AS average_salary
FROM
    employees
GROUP BY
    department
"""
explanation = "Show the average salary for each department."

vn.train(
    df=pd.DataFrame({'department': ['Sales', 'Engineering', 'Marketing', 'HR'], 'average_salary': [
                    55000, 75000, 60000, 65000]}),
    sql=chart_sql,
    question=explanation
)
# 8. Ask Questions and Get SQL
#    -  Now, we can ask questions in natural language.
#    -  Vanna will generate the corresponding SQL query.

question1 = "What is the average salary per department?"
sql1 = vn.ask(question1)
print(f"Question: {question1}")
print(f"Generated SQL: {sql1}")

question2 = "Show me the names of employees who earn more than 60000"
sql2 = vn.ask(question2)
print(f"Question: {question2}")
print(f"Generated SQL: {sql2}")

# 9. Execute the SQL and Get Results
#     -  Execute the generated SQL queries against the database.
#     -  Vanna doesn't do this automatically, you need to use your database connection.


def execute_query(conn, query):
    """Executes a SQL query and returns the result as a Pandas DataFrame."""
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


print("\nResults:")
df1 = execute_query(conn, sql1)
if df1 is not None:
    print(df1)

df2 = execute_query(conn, sql2)
if df2 is not None:
    print(df2)

# 10. (Optional)  Retrieve information from the vector store
print("\nInformation stored in Vector Store:")
print(vn.get_training_data())

# 11.  (Optional) Delete training data.
# vn.remove_training_data(question=question1)
# print("\nInformation stored in Vector Store after deleting the first question:")
# print(vn.get_training_data())
