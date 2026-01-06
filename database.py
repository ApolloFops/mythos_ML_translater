import sqlite3

import discord

from .config import DATABASE_PATH


class TranslationDatabaseQueries:
	create_translation_table = """
CREATE TABLE IF NOT EXISTS translations (
	message_text TEXT NOT NULL CHECK (message_text <> ''),
	channel_id TEXT,
	message_id TEXT,
	translation TEXT
)
"""

	check_for_translation = """
SELECT EXISTS(SELECT 1 FROM translations WHERE message_id = "{message_id}" LIMIT 1);
"""

	write_message = """
INSERT INTO
	translations (message_text, channel_id, message_id)
VALUES
	(?, ?, ?);
"""

	find_random_untranslated = """
SELECT channel_id, message_id, message_text
FROM translations
WHERE (translation IS NULL OR translation = '')
AND channel_id IN ({parameter_placeholders})
ORDER BY RANDOM()
LIMIT 1
"""

	update_translation = """
UPDATE translations
SET translation = ?
WHERE message_id = ?
"""

	update_channel_id = """
UPDATE translations
SET channel_id = ?
WHERE message_id = ?
"""


class TranslationDatabase:
	def __init__(self):
		# Make sure the table exists in the database
		with self.connect_db() as db:
			self.exec_db(db, TranslationDatabaseQueries.create_translation_table)

	def connect_db(self):
		return sqlite3.connect(DATABASE_PATH)

	def exec_db(self, connection: sqlite3.Connection, query: str):
		connection.cursor().execute(query)
		connection.commit()

	def read_db(self, connection: sqlite3.Connection, query: str):
		return connection.cursor().execute(query).fetchone()

	def check_for_translation(self, message_id):
		# Make sure the message hasn't already been starboarded
		with self.connect_db() as db:
			in_database = self.read_db(db, TranslationDatabaseQueries.check_for_translation.format(message_id=str(message_id)))[0]

		return str(in_database) == "1"

	def add_message(self, message: discord.Message):
		with self.connect_db() as db:
			# There might be a better way to do this but this works
			db.cursor().execute(TranslationDatabaseQueries.write_message, (message.content, str(message.channel.id), str(message.id)))

			db.commit()

	def get_random_untranslated(self, channel_filter):
		placeholders = ",".join("?" for _ in channel_filter)

		with self.connect_db() as db:
			return db.cursor().execute(TranslationDatabaseQueries.find_random_untranslated.format(parameter_placeholders=placeholders), list(map(str, channel_filter))).fetchone()

	def update_translation(self, message_id, translation):
		with self.connect_db() as db:
			db.cursor().execute(TranslationDatabaseQueries.update_translation, (translation, message_id))

	def update_channel_id(self, message_id, channel_id):
		with self.connect_db() as db:
			db.cursor().execute(TranslationDatabaseQueries.update_translation, (channel_id, message_id))
