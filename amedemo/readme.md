## LensXPlain: Visualizing and Explaining Contributing Subsets for Aggregate Query Answers

### Required packages
Backend
- Flask_SQLAlchemy
- numpy
- SQLAlchemy
- psycopg2
- efficient_apriori
- pandas
- Flask
- scikit_learn

See requirements.txt

Frontend
- D3.js
- Vega
- Vega-Lite
- jQuery
- Bootstrap.js

### How to run?
#### 0. Install dependencies.

You can run ```pip install -r requirements.txt``` to install all dependencies.

#### 1. Create and configure your database

Currently LensXplain runs on top of Postgresql. After setting up your database, you can specify your database configuration in ```config.py```

#### 2. Run the application

Inside the LensXplain directory, run
```
export FLASK_APP=runserver.py
flask run
```

Then open ```localhost:5001``` in your browser.
