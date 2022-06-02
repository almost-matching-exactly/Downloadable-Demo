import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, "/var/www/zm42/grading_demo")
from ra_test_demo.app import app as application
