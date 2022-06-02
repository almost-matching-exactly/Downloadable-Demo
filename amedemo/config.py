
#SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:hj112@ame-demo.cs.duke.edu/postgres'
SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:leefs100@localhost/postgres'



DB_NAME = 'postgres'
SQLALCHEMY_BINDS = {
    #'raw': 'postgresql+psycopg2://postgres:hj112@ame-demo.cs.duke.edu/postgres'
    'raw': 'postgresql+psycopg2://postgres:leefs100@localhost/postgres'


}
SQLALCHEMY_POOL_SIZE = 10
SQLALCHEMY_TRACK_MODIFICATIONS = False
JSONIFY_PRETTYPRINT_REGULAR = False
