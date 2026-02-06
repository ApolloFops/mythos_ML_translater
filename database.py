import sqlite3

import discord
from typing import List
from datetime import datetime

from .config import DATABASE_PATH


class TranslationDatabaseQueries:
	# Translations
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

	get_current_translated_count = """
SELECT COUNT(*) FROM translations WHERE translation IS NOT NULL AND translation != '';
"""

	# Metadata
	create_metadata_table = """
CREATE TABLE IF NOT EXISTS metadata (
	id INTEGER PRIMARY KEY CHECK (id = 1),
	last_train_message_count INTEGER NOT NULL DEFAULT 0,
	last_train_date TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

	insert_initial_metadata = """
INSERT OR IGNORE INTO metadata (id) VALUES (1);
"""

	get_new_translations_since_last_train = """
SELECT
	(SELECT COUNT(*) FROM translations WHERE translation IS NOT NULL AND translation != '') -
	last_train_message_count
FROM metadata WHERE id = 1;
"""

	get_last_train_date = """
SELECT last_train_date FROM metadata WHERE id = 1;
"""

	update_metadata_after_train = """
UPDATE metadata
SET last_train_message_count = (
		SELECT COUNT(*) FROM translations WHERE translation IS NOT NULL AND translation != ''
	),
	last_train_date = datetime('now')
WHERE id = 1;
"""


class TranslationDatabase:
	def __init__(self):
		# Make sure the table exists in the database
		with self.connect_db() as db:
			db.cursor().execute(TranslationDatabaseQueries.create_metadata_table)
			db.cursor().execute(TranslationDatabaseQueries.insert_initial_metadata)
			db.cursor().execute(TranslationDatabaseQueries.create_translation_table)
			db.commit()

	def connect_db(self):
		return sqlite3.connect(DATABASE_PATH)

	def read_db(self, connection: sqlite3.Connection, query: str):
		return connection.cursor().execute(query).fetchone()

	def check_for_translation(self, message_id):
		# Make sure the message hasn't already been starboarded
		with self.connect_db() as db:
			in_database = self.read_db(db, TranslationDatabaseQueries.check_for_translation.format(message_id=str(message_id)))[0]

		return str(in_database) == "1"

	def add_messages(self, messages: List[discord.Message]):
		with self.connect_db() as db:
			for message in messages:
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

	def get_current_translated_count(self):
		with self.connect_db() as db:
			return self.read_db(db, TranslationDatabaseQueries.get_current_translated_count)[0]

	def get_new_translations_since_last_train(self):
		with self.connect_db() as db:
			return self.read_db(db, TranslationDatabaseQueries.get_new_translations_since_last_train)[0]

	def get_last_train_date(self):
		with self.connect_db() as db:
			timestamp = self.read_db(db, TranslationDatabaseQueries.get_last_train_date)[0]
			# SQLite datetime('now') uses UTC by default
			return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

	def update_metadata_after_train(self):
		with self.connect_db() as db:
			db.cursor().execute(TranslationDatabaseQueries.update_metadata_after_train)
			db.commit()
