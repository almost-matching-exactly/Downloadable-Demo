import tempfile
import sys
import re
import psycopg2
import datetime
import logging


class QueryRunner:

    def __init__(self, sql_query, db_raw):
        self.sql_query = sql_query
        self.error_message = None
        self.cur = db_raw

    def evaluate_query(self):

        try:
            # self.cur.execute(q_diff)
            # return self.cur.fetchall()
            return self.cur.execute(self.sql_query).fetchall()
        except psycopg2.Error as e:
            self.error_message = 'There is something wrong with your query!'
            return e
