import sqlite3

from .config import DATABASE_PATH

class TranslationDatabase:
	create_translation_table = """
CREATE TABLE IF NOT EXISTS translations (
	message_text TEXT NOT NULL,
	message_id TEXT,
	translation TEXT
)
"""

	def __init__(self):
		# Make sure the table exists in the database
		with self.connect_db() as db:
			self.exec_db(db, self.create_translation_table)

	def connect_db(self):
		return sqlite3.connect(DATABASE_PATH)

	def exec_db(self, connection: sqlite3.Connection, query: str):
		connection.cursor().execute(query)
		connection.commit()

	def read_db(self, connection: sqlite3.Connection, query: str):
		return connection.cursor().execute(query).fetchone()

