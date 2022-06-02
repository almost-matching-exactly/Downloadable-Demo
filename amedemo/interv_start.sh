#!/bin/bash
if test -f "package_install.log";
then echo "Package already installed."
else
    pip install -r requirements.txt 
    echo "Package Installed."> package_install.log
    echo "Package Installed."
fi

CONFIG_FILE=config.py
if test -f "$CONFIG_FILE"; 
then
    echo "$CONFIG_FILE exists."
else
    echo "$CONFIG_FILE not exist!"
    echo "Postgres URL:"
    echo "Example: postgresql+psycopg2://username:password@localhost:port/dbname"
    read DB_URL
    # echo $DB_URL
    echo "DB_NAME"
    read DB_NAME
    # echo $DB_NAME

    echo "SQLALCHEMY_DATABASE_URI = '$DB_URL'" > $CONFIG_FILE
    echo "DB_NAME = '$DB_NAME'" >>  $CONFIG_FILE
    echo "SQLALCHEMY_BINDS = {" >>  $CONFIG_FILE
    echo "    'raw': '$DB_URL'" >>  $CONFIG_FILE
    echo "}" >>  $CONFIG_FILE
    echo "SQLALCHEMY_POOL_SIZE = 10" >>  $CONFIG_FILE
    echo "SQLALCHEMY_TRACK_MODIFICATIONS = False" >>  $CONFIG_FILE
fi

# Uncomment this line for starting Postgres
# pg_ctl -D /usr/local/var/postgres start

python runserver.py

