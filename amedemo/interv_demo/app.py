# -*- coding: utf-8 -*-

from operator import truediv
from re import M
from tkinter.messagebox import NO

#loginstuff
from flask import Flask, current_app, abort, session

from flask import Markup
from flask import render_template, request, jsonify, redirect, url_for, flash
import sqlalchemy.pool as pool
from flask_sqlalchemy import SQLAlchemy
import datetime
#import logging
import traceback
import os
import csv
import io
import pandas as pd
import dame_flame  
import sys 

import simplejson as json
# import json
from interv_backend import QueryRunner, IntervFinder, AttributeRecommender

import config as cf

import numpy as np
from numpy import int0, int32, mat
#import psycopg2
#from psycopg2.extensions import register_adapter, AsIs
#psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
from pandas.core.frame import DataFrame

#loginstuff
from flask_session import Session


#logging stuff
from flask import Response


db_raw = None
db_prov = None
db_reev = None
db_log = None
available_tables = None
table_schemas = None
table_datatype = None
active_table = None
table_numeric_attr = None
flame_model = None
myname = None
treat_var = None
out_var = None
logoutput = None
result_dataframe = None
current_entry_id = None
current_mmg = None
curr_path = None
ate = None
att = None


app = Flask(__name__, static_folder='static')
app.secret_key = 'secretkeyhereplease'
app.config.from_pyfile('../config.py')
#gunicorn_logger = logging.getLogger('gunicorn.error')
#app.logger.handlers = gunicorn_logger.handlers
# app.logger.setLevel(gunicorn_logger.level)


# login bullshit 1
#app.config["SESSION_PERMANENT"] = False
#app.config["SESSION_TYPE"] = "filesystem"

#Session(app)




#test123



db = SQLAlchemy(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# DB_NAME = 'postgres'
# SQLALCHEMY_DATABASE_URI = cf.SQLALCHEMY_DATABASE_URI.split('//')[0]+'//' + cf.SQLALCHEMY_DATABASE_URI.split('/')[-2] + '/' + DB_NAME
# # SQLALCHEMY_BINDS = {
# #     'raw': SQLALCHEMY_DATABASE_URI
# # }
# SQLALCHEMY_BINDS = cf.SQLALCHEMY_BINDS
# SQLALCHEMY_BINDS['new'] = SQLALCHEMY_DATABASE_URI


@app.errorhandler(404)
def page_not_found(e):
    flash("Page not found! Redirecting...")
    return redirect('/flame')


@app.errorhandler(500)
def internal_server_error(e):
    flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
    return redirect('/flame')


@app.route('/switch_db/<db_name>', methods=['GET'])
def switch_db(db_name='postgres'):
    SQLALCHEMY_DATABASE_URI = cf.SQLALCHEMY_DATABASE_URI.split(
        '//')[0]+'//' + cf.SQLALCHEMY_DATABASE_URI.split('/')[-2] + '/' + db_name
    SQLALCHEMY_BINDS = cf.SQLALCHEMY_BINDS
    SQLALCHEMY_BINDS['new'] = SQLALCHEMY_DATABASE_URI
    app.config.from_mapping({'SQLALCHEMY_BINDS': SQLALCHEMY_BINDS})
    return interv()


@app.route('/', methods=['GET'])
@app.route('/flame', methods=['GET'])
@app.route('/flame/<active_table>', methods=['GET'])
# @login_required
def interv(active_table='adult'):
    """Renders the test page."""

    # with app.app_context():
    #     db.init_app(app)

    # app = Flask(__name__, static_folder='static')
    # app.secret_key = 'secretkeyhereplease'

    # db = SQLAlchemy(app)
    # APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    # init_db()
    # app.run()


    #session["name"] = None




    # get available tables
    try:
        globals()['db_raw'] = db.get_engine(app, bind='new')
    except:
        globals()['db_raw'] = db.get_engine(app, bind='raw')
    globals()['attr_rmd'] = AttributeRecommender.AttributeRecommender(db_raw)
    cursor = db_raw




    # kehan modified for select all db names
    db_query = 'SELECT datname FROM pg_database;'

    globals()['available_dbs'] = list(map(
        lambda x: x[0],
        cursor.execute(db_query).fetchall()
    ))

    print(globals()['available_dbs'])

    table_query = '''
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_schema, table_name;
    '''
    globals()['available_tables'] = list(map(
        lambda x: x[0],
        cursor.execute(table_query).fetchall()
    ))
    if 'ra_users' in globals()['available_tables']:
        globals()['available_tables'].remove('ra_users')
    # globals()['available_tables'] = ['adult']

    # get schema for each table
    globals()['table_schemas'] = {}
    globals()['table_datatype'] = {}
    globals()['table_numeric_attr'] = {}
    for tbl in globals()['available_tables']:
        globals()['table_schemas'][tbl] = cursor.execute(
            "select * FROM {} LIMIT 1;".format(tbl)).keys()
        globals()['table_datatype'][tbl] = cursor.execute('''
                    select column_name, data_type 
                    from information_schema.columns 
                    where table_name = '{}';
                    '''.format(tbl)).fetchall()
        globals()['table_numeric_attr'][tbl] = cursor.execute('''
            select column_name 
            from information_schema.columns 
            where table_name = '{}'
            and data_type in ('smallint', 'integer', 'bigint', 
                        'decimal', 'numeric', 'real', 'double precision',
                        'smallserial', 'serial', 'bigserial', 'money');
            '''.format(tbl)).fetchall()



    globals()['active_table'] = active_table


            #login bullshit 2
    #if not session.get("name"):
    #    print("not session")

    #    return redirect("/login")




    print("rendering interv html")

    return render_template(
        'interv.html',
        ra_result=None,
        cur_user=request.environ.get('REMOTE_USER'),
        available_dbs=available_dbs,
        active_table=active_table,
        available_tables=globals()['available_tables'],
        num_table=6,
        table_schemas=table_schemas,
        table_datatype=table_datatype,
        table_numeric_attr=globals()['table_numeric_attr']

    )



@app.route('/gen_data_open/', methods=['GET', 'POST'])
def gen_data_open(): 
    return render_template('gen_data_open.html',
        ra_result=None,
        cur_user=request.environ.get('REMOTE_USER'),
        available_dbs=available_dbs,
        active_table=active_table,
        available_tables=globals()['available_tables'],
        num_table=6,
        table_schemas=table_schemas,
        table_datatype=table_datatype,
        table_numeric_attr=globals()['table_numeric_attr']
)







#logging stuff???
# configure logger
#logger.add("app/static/job.log", format="{time} - {message}")

# adjusted flask_logger
def flask_logger():
    """creates logging information"""
    with open("app/static/job.log") as log_info:
        #for i in range(25):
            #logger.info(f"iteration #{i}")
        data = log_info.read()
        yield data.encode()
            #sleep(1)
        # Create empty job.log, old logging will be deleted
        #open("app/static/job.log", 'w').close()

        #log_info.seek(0,2) # Go to the end of the file
        #while True:
        #    line = log_info.readline()
        #    if not line:
        #        time.sleep(0.01) # Sleep briefly
        #        continue
        #    yield line.encode()


@app.route("/log_stream", methods=["GET"])
def log_stream():
    """returns logging information"""    
    return Response(flask_logger(), mimetype="text/plain", content_type="text/event-stream")






def alg_predict(df, treat_var, out_var, alg, alpha, repeats, early_stop_iterations, stop_unmatched_c, stop_unmatched_t, early_stop_un_c_frac, 
    early_stop_un_t_frac, early_stop_pe, early_stop_pe_frac, missing_holdout_replace, missing_data_replace, missing_holdout_imputations, missing_data_imputations):


        
    #if os.path.exists("app/static/job.log"):
    #    os.remove("app/static/job.log")

    #log = open("app/static/job.log", "w")

    #with open("app/static/job.log", "w") as log: 
        #sys.stdout = log

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout


    #if alg=="dame":
    #    model = dame_flame.matching.DAME(alpha=alpha, verbose=3, repeats=repeats, early_stop_iterations=early_stop_iterations,
    #        stop_unmatched_c=stop_unmatched_c, stop_unmatched_t=stop_unmatched_t, early_stop_un_c_frac=early_stop_un_c_frac, early_stop_un_t_frac=early_stop_un_t_frac,
    #        early_stop_pe=early_stop_pe, early_stop_pe_frac=early_stop_pe_frac, want_pe=True, want_bf=True, missing_holdout_replace=missing_holdout_replace, 
    #        missing_data_replace=missing_data_replace, missing_holdout_imputations=missing_holdout_imputations, 
    #        missing_data_imputations=missing_data_imputations)
    #else:
    model = dame_flame.matching.FLAME(alpha=alpha, verbose=3, repeats=repeats, early_stop_iterations=early_stop_iterations,
        stop_unmatched_c=stop_unmatched_c, stop_unmatched_t=stop_unmatched_t, early_stop_un_c_frac=early_stop_un_c_frac, early_stop_un_t_frac=early_stop_un_t_frac,
        early_stop_pe=early_stop_pe, early_stop_pe_frac=early_stop_pe_frac, want_pe=True, want_bf=True, missing_holdout_replace=missing_holdout_replace, 
        missing_data_replace=missing_data_replace, missing_holdout_imputations=missing_holdout_imputations, 
        missing_data_imputations=missing_data_imputations)

    model.fit(df, treatment_column_name=treat_var, outcome_column_name=out_var)
    myres = model.predict(df)

    logoutput = new_stdout.getvalue()
    
    globals()['logoutput'] = logoutput


    #sys.stdout = sys.__stdout__
    #print('test two')

    #sys.stdout = old_stdout


    return model, myres










@app.route('/run_query', methods=['POST'])
# @login_required
def run_query():

    form_data = request.form
    app.logger.debug(form_data)

    globals()['db_raw'] = db.get_engine(app, bind='raw')
    cursor = db_raw
    sql_query = '''SELECT * FROM {}'''.format(
        form_data['sql-from']
    )
    myresult = cursor.execute(sql_query).fetchall()

    sql_query_two = '''SELECT column_name FROM information_schema.columns WHERE table_name = N'{}' '''.format(
        form_data['sql-from']
    )
    mycols = cursor.execute(sql_query_two).fetchall()


    cols = []
    for col in mycols:
        cols.append(col[0])
    df = pd.DataFrame(myresult, columns=cols)

    treat_var = request.form.get('xvar')
    out_var = request.form.get('y-var')

    globals()['treat_var'] = treat_var
    globals()['out_var'] = out_var

    #session['treat_var'] = treat_var
    #session['out_var'] = out_var


    my_size = len(df.index)





    #error bullshit: check if user selected same var as treat and outcome 
    if treat_var == out_var: 
        flash("do not select same variable as treatment and outcome")
        return app.make_response(("do not select same variable as treatment and outcome", 500))


    if form_data['exclude'] != "":
        exclude_vars = form_data['exclude'].split(',')
        df = df.drop(columns=exclude_vars)




    #error bullshit: check if user dropped treatment and/or outcome: 
    if treat_var not in df or out_var not in df:
        flash("selected treatment and/or outcome variable not in data. check that selected treatment and/or outcome variables were not excluded.")
        return app.make_response(("selected treatment and/or outcome variable not in data. check that selected treatment and/or outcome variables were not excluded.", 500))

    elif df.loc[:, ~df.columns.isin([treat_var, out_var])].empty:
        flash("data other than treatment and outcome variables missing. do not drop all non-treatment and non-outcome variables")
        return app.make_response(("data other than treatment and outcome variables missing. do not drop all non-treatment and non-outcome variables", 500))

    

    treatment_condition = form_data["treat_condition"]

    #error bullshit: treatment condition syntax
    if treatment_condition !="" and treatment_condition[0] != 'x':
        flash("please input treatment condition with syntax x==[some python-compliant boolean expression].")
        return app.make_response(("please input treatment condition with syntax x==[some python-compliant boolean expression].", 500))

    if treatment_condition != "":
        #treatment_condition = "x"+treatment_condition
        df[treat_var] = df[treat_var].apply(lambda x: 1 if eval(treatment_condition) else 0)
        treatment_condition = treatment_condition[1:]

    binary_cols = df.columns[df.isin([0,1]).all()].tolist()
    if treat_var not in binary_cols:
        flash("treatment variable must be binary")
        return app.make_response(("treatment variable must be binary", 500))

    
    bool_params = [form_data['repeats'], form_data['sumc'], form_data['sumt'], form_data['espe']]
    for j in range(4):
        if bool_params[j]=="False":
            bool_params[j] = False
        else:
            bool_params[j] = True
    #adaptive_weights=form_data['aweights'],
    
    alg = form_data['algorithm']




    #if alg=="dame":
    #    model = dame_flame.matching.DAME(alpha=float(form_data['alpha']), verbose=3, repeats=bool_params[0], early_stop_iterations=int(form_data['esi']),
    #        stop_unmatched_c=bool_params[1], stop_unmatched_t=bool_params[2], early_stop_un_c_frac=float(form_data['esucf']), early_stop_un_t_frac=float(form_data['esutf']),
    #        early_stop_pe=bool_params[3], early_stop_pe_frac=float(form_data['espf']), missing_holdout_replace=int(form_data['missing-holdout-replace']), 
    #        missing_data_replace=int(form_data['missing-data-replace']), missing_holdout_imputations=int(form_data['missing-holdout-imputations']), 
    #        missing_data_imputations=int(form_data['missing-data-imputations']))
    #else:
    #    model = dame_flame.matching.FLAME(alpha=float(form_data['alpha']), repeats=bool_params[0], verbose=3, early_stop_iterations=int(form_data['esi']),
    #        stop_unmatched_c=bool_params[1], stop_unmatched_t=bool_params[2], early_stop_un_c_frac=float(form_data['esucf']), early_stop_un_t_frac=float(form_data['esutf']),
    #        early_stop_pe=bool_params[3], early_stop_pe_frac=float(form_data['espf']), missing_holdout_replace=int(form_data['missing-holdout-replace']), 
    #        missing_data_replace=int(form_data['missing-data-replace']), missing_holdout_imputations=int(form_data['missing-holdout-imputations']), 
    #        missing_data_imputations=int(form_data['missing-data-imputations']))

    #model.fit(df, treatment_column_name=treat_var, outcome_column_name=out_var)
    #myres = model.predict(df)



    model, myres = alg_predict(df=df, treat_var=treat_var, out_var=out_var, alg=alg, alpha=float(form_data['alpha']), repeats=bool_params[0], 
        early_stop_iterations=int(form_data['esi']), stop_unmatched_c=bool_params[1], stop_unmatched_t=bool_params[2], early_stop_un_c_frac=float(form_data['esucf']), 
        early_stop_un_t_frac=float(form_data['esutf']), early_stop_pe=bool_params[3], early_stop_pe_frac=float(form_data['espf']), 
        missing_holdout_replace=int(form_data['missing-holdout-replace']), missing_data_replace=int(form_data['missing-data-replace']), 
        missing_holdout_imputations=int(form_data['missing-holdout-imputations']), missing_data_imputations=int(form_data['missing-data-imputations']))

    
    globals()['flame_model']=model



    result_flame = myres.replace(to_replace='*', value=np.nan)



    #globals()['flame_model']=model
    #session['name'] = model

    mytreated = []
    myoutcome = []




    too_large = 0
    if (len(myres.index)>=5000):
        myres = myres.head(1000)
        too_large = 1


    myrows = list(myres.index)
    for r in myrows:
        mytreated.append(df.iloc[r][treat_var])
        myoutcome.append(df.iloc[r][out_var])

    myres[treat_var + " (treatment)"] = mytreated
    myres[out_var + " (outcome)"] = myoutcome

    myres = myres[[treat_var + " (treatment)", out_var + " (outcome)"] + [c for c in myres if c not in [treat_var + " (treatment)", out_var + " (outcome)"]]]

    globals()['result_dataframe'] = myres




    test_res = list(myres.to_records(index=True))

    my_attributes = list(myres.columns.values)
    my_attributes.insert(0,'index')

    #final_result = [dict(row) for row in test_result]
    final_res = []
    for row in test_res:
        count = 0
        rowentry = {}
        for i in row:
            rowentry[my_attributes[count]] = i
            if i != "*":
                rowentry[my_attributes[count]] = str(rowentry[my_attributes[count]])
            count +=1
        final_res.append(rowentry)



    if myres.empty:
        ate = "none"
        att = "none"
        no_matches = 1
    else:
        ate = str(dame_flame.utils.post_processing.ATE(matching_object=model))
        att = str(dame_flame.utils.post_processing.ATT(matching_object=model))
        no_matches = 0

    globals()['ate'] = ate
    globals()['att'] = att



    group_size_treated = []
    group_size_overall = []
    cate_of_group = []
    for group in model.units_per_group:
        
        # find len of just treated units
        df_mmg = df.loc[group]
        treated = df_mmg.loc[df_mmg[treat_var] == 1]
        
        group_size_treated.append(len(treated))
        group_size_overall.append(len(group))
        
        cate_of_group.append(dame_flame.utils.post_processing.CATE(model, group[0]))

    
    covar_matches = result_flame.count(axis=0)
    covar_matches.to_frame()

    rnm = covar_matches.index
    rownames = list(rnm)

    covar_count = covar_matches.values.tolist()

    

    #sql_query = '''SELECT {}, {} FROM {}{} GROUP BY {} ;'''.format(
    #    form_data['sql-select'],
    #    form_data['sql-aggregate'],
    #    form_data['sql-from'],
    #    (' WHERE ' + form_data['sql-add-where'] if 'sql-add-where' in form_data and
    #                                               len(form_data['sql-add-where'].strip()) > 0 else ''),
    #    form_data['sql-select']
    #)
    #groupby_attributes = list(
    #    map(lambda x: x.strip(), form_data['sql-select'].split(',')))

    # app.logger.debug('Running query: {}'.format(str(sql_query)))
    # return query result
    try:
        #qr = QueryRunner.QueryRunner(sql_query, db_raw)
        #qr_result = list(map(
        #    lambda x: dict(zip(groupby_attributes + [form_data['sql-aggregate']],
        #                       list(map(lambda y: y.strip() if isinstance(y, str) else y,
        #                                x)))),
        #    qr.evaluate_query()))

        #print(qr_result)

        #if qr.error_message is not None:
        #    flash(qr_result)
        #    return app.make_response(('', 500))
        #else:

        return jsonify(status_code=200,
                           #query_result=qr_result,
                           #groupby_attributes=groupby_attributes,
                           #aggregation=form_data['sql-aggregate'],
                           
                           logoutput=globals()['logoutput'],
                           treat_var=treat_var + " " + treatment_condition,
                           out_var=out_var,
                           ate=ate,
                           att=att,
                           too_large=too_large,
                           no_matches=no_matches,
                           result=final_res,
                           attributes=my_attributes,
                           group_size_treated=group_size_treated,
                           cate_of_group=cate_of_group,
                           rownames=rownames,
                           covar_count=covar_count)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
        return app.make_response((str(e), 500))




@app.route('/gen_data', methods=['POST'])
def gen_data(): 

    form_data = request.form

    print(form_data)


    gen_type = form_data.get("gen_type")

    c_i = []
    if gen_type != "bidec": 
        if gen_type=="uniform": 
            covar_import = form_data.get("covar_import_u")
            num_cov = int(form_data.get("num_covar_u"))
        else: 
            covar_import = form_data.get("covar_import_b")
            num_cov = int(form_data.get("num_covar_b"))

        
        cov_im = covar_import.split(',')
        for ci in cov_im: 
            c_i.append(float(ci))

        if len(c_i) != num_cov: 
            flash("Please list exactly 1 importance per covariate, such that this entry is the same length as the number of covariates. ")
            return app.make_response(("Please list exactly 1 importance per covariate, such that this entry is the same length as the number of covariates. ", 500))




    if gen_type=="uniform": 
        df = dame_flame.utils.data.generate_uniform_given_importance(num_control=int(form_data.get("num_control_u")), num_treated=int(form_data.get("num_treat_u")),
                                    num_cov=int(form_data.get("num_covar_u")), min_val=int(form_data.get("min_val_u")),
                                    max_val=int(form_data.get("max_val_u")), covar_importance=c_i,
                                    bi_mean=float(form_data.get("bi_mean_u")), bi_stdev=float(form_data.get("bi_stdev_u")))

    elif gen_type=="binom": 
        df = dame_flame.utils.data.generate_binomial_given_importance(num_control=int(form_data.get("num_control_b")), num_treated=int(form_data.get("num_treat_b")),
                                num_cov=int(form_data.get("num_covar_b")), bernoulli_param=float(form_data.get("bernoulli_param_b")),
                                bi_mean=float(form_data.get("bi_mean_b")), bi_stdev=float(form_data.get("bi_stdev_b")),
                                covar_importance=c_i)
    else: 

        #nc = int(form_data.get("num_control"))
        nt = int(form_data.get("num_treat"))
        num_cov = int(form_data.get("num_covar"))

        


        df = dame_flame.utils.data.generate_binomial_decay_importance(num_control=int(form_data.get("num_control")), num_treated=int(form_data.get("num_treat")),
                                  num_cov=int(form_data.get("num_covar")), bernoulli_param=float(form_data.get("bernoulli_param")),
                                  bi_mean=float(form_data.get("bi_mean")), bi_stdev=float(form_data.get("bi_stdev")))

    if not form_data.get("dname"): 
        dname = "uniform"
    elif gen_type=="uniform": 
        dname = form_data.get("dname_u")
    elif gen_type=="binom": 
        dname = form_data.get("dname_b")
    else: 
        dname = form_data.get("dname")

    path = "gen_data/"

    if not os.path.exists(path):
        os.mkdir(path)

    if dname + '.csv' in os.listdir(path):
        i = 1
        mydname = dname + "(" + str(i) + ")"
        while mydname + '.csv' in os.listdir(path):
            i += 1
            mydname = dname + "(" + str(i) + ")"
                
        dname = mydname


    df[0].to_csv(path + dname + '.csv', index=True)

    try:
        return jsonify(status_code=200)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
        return app.make_response((str(e), 500))










@app.route('/download_result', methods=['POST'])
def download_result(): 
    form_data = request.form

    myres = globals()['result_dataframe']
    mmg = globals()['current_mmg']
    current_id = str(globals()['current_entry_id'])

    curr_match = form_data['download-name']
    if curr_match=="":
        curr_match = form_data['sql-from'] + '_result'


    path = 'results/' + curr_match


    if (form_data['download-type']=='general'):

        if not myres.empty: 

            if os.path.exists(path):
                i = 1
                mypath = path + "(" + str(i) + ")"
                while os.path.exists(mypath):
                    i += 1
                    mypath = path + "(" + str(i) + ")"
                
                path = mypath
                


                #os.mkdir(path)
            globals()['curr_path'] = path
            os.mkdir(path)

            download_path = path + '/result.csv'
            myres.to_csv(download_path, index=True)


            missing_holdout_replace = ["assume no missing holdout data", "", "exclude units with missing values", "MICE"]
            missing_data_replace = ["assume no missing data", "do not match on units with missing values", "prevent all missing values from matching", "MICE"]

            f = open(path + '/parameters.txt', 'w')
            f.write("treatment variable = " + form_data['xvar'])
            f.write("\noutcome variable = " + form_data['y-var'])
            if form_data['treat_condition'] != "": 
                f.write("\ntreatment condition: treatment variable = 1 if " + form_data['xvar'] + form_data['treat_condition'][1:])
            f.write("\n")
            f.write("\nAdvanced settings: ")
            f.write("\nAlpha = " + str(form_data['alpha']))
            f.write("\nRepeats = " + form_data['repeats'])
            f.write("\nEarly stop iterations = " + form_data['esi'])
            f.write("\nStop unmatched control = " + form_data['sumc'])
            f.write("\nStop unmatched treatment = " + form_data['sumt'])
            f.write("\nEarly stop fraction for control units = " + form_data['esucf'])
            f.write("\nEarly stop fraction for treatment units = " + form_data['esutf'])
            f.write("\nEarly stop for PE = " + form_data['espe'])
            f.write("\nEarly stop fraction for PE = " + form_data['espf'])
            f.write("\nMissing holdout replace: " + missing_holdout_replace[int(form_data['missing-holdout-replace'])])
            if form_data['missing-holdout-replace']==3: 
                f.write("\nMissing holdout imputations = " + form_data['missing-holdout-imputations'])
            f.write("\nMissing data replace: " + missing_data_replace[int(form_data['missing-data-replace'])])
            if form_data['missing-data-replace']==3: 
                f.write("\nMissing data imputations = " + form_data['missing-data-imputations'])

            f.close()

            fi = open(path + '/ate.txt', 'w')
            fi.write("Average Treatment Effect = " + str(globals()['ate']))
            fi.write("\nAverage Treatment Effect on the Treated = " + str(globals()['att']))
            fi.close()

    
    else: 

        path = globals()['curr_path']

        mmg_path = path + '/matched_groups'

        if not os.path.exists(mmg_path):
            #mmg_directory = '/matched_groups'
            #mmg_path = os.path.join(path, mmg_directory)
            os.mkdir(mmg_path)

        if not mmg.empty: 
            download_path = mmg_path + '/unit' + current_id + "_mmg.csv"
            mmg.to_csv(download_path, index=True)

    



    try:
        return jsonify(status_code=200)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
        return app.make_response((str(e), 500))






@app.route('/get_mmg', methods=['POST'])
# @login_required
def get_mmg():

    form_data = request.form

    entry_id = int(form_data['entry-id'])
    globals()['current_entry_id'] = entry_id


    mmg = dame_flame.utils.post_processing.MG(matching_object=flame_model, unit_ids=entry_id)



    mg_too_large = 0
    no_match = 0
    final_mg = []
    my_attributes = []
    cate = "none"

    if not isinstance(mmg, DataFrame):
        no_match = 1
        globals()['current_mmg'] = pd.DataFrame()
    else: 

        if (len(mmg.index)>=5000):
            mmg = mmg.head(1000)
            mg_too_large = 1


        #cate = dame_flame.utils.post_processing.CATE(matching_object=flame_model, unit_ids=entry_id)

        no_match_cols = []

        for col in mmg.columns:
            if mmg.iloc[0][col] == "*":
                no_match_cols.append(col)

        mmg = mmg[[c for c in mmg if c not in no_match_cols] + no_match_cols]
        mmg = mmg[[globals()['treat_var'], globals()['out_var']] + [c for c in mmg if c not in [globals()['treat_var'], globals()['out_var']]]]

        globals()['current_mmg'] = mmg



        #loginstuff
        #cate = dame_flame.utils.post_processing.CATE(matching_object=session["name"], unit_ids=entry_id)
        cate = dame_flame.utils.post_processing.CATE(matching_object=flame_model, unit_ids=entry_id)


        hold_mg = list(mmg.to_records(index=True))

        my_attributes = list(mmg.columns.values)
        my_attributes.insert(0,'index')


        for row in hold_mg:
            count = 0
            rowentry = {}
            for i in row:
                rowentry[my_attributes[count]] = i
                if i != "*":
                    rowentry[my_attributes[count]] = str(rowentry[my_attributes[count]])
                count +=1
            final_mg.append(rowentry)

    #loginstuff
    #print("mmg session = " + str(session["name"]))


    try:
        return jsonify(status_code=200,
                        mg_too_large=mg_too_large, 
                        matched_group=final_mg,
                        attributes=my_attributes, 
                        no_match=no_match,
                        cate=str(cate))
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
        return app.make_response((str(e), 500))



@app.route('/drop_var', methods=['POST'])
# @login_required
def drop_var():

    form_data = request.form

    globals()['db_raw'] = db.get_engine(app, bind='raw')
    cursor = db_raw

    features_to_drop = form_data['drop']
    #features_to_drop = features_to_drop.replace(',', ', ')

    features = features_to_drop.split(',')

    sql_query = '''ALTER TABLE {} '''.format(
        form_data['table']
    )

    for feature in features: 
        sql_add = '''DROP COLUMN {}, '''.format(feature)
        sql_query = sql_query + sql_add


    sql_query = sql_query[:len(sql_query) - 2]
    print(sql_query)


    #sql_query = '''ALTER TABLE {} DROP COLUMN {}'''.format(
    #    form_data['table'], 
    #    features_to_drop
    #)

    cursor.execute(sql_query)



    try:
        return jsonify(status_code=200)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
        return app.make_response((str(e), 500))


@app.route('/drop_table', methods=['POST'])
# @login_required
def drop_table():

    form_data = request.form

    globals()['db_raw'] = db.get_engine(app, bind='raw')
    cursor = db_raw

    tables_to_drop = form_data['table_drop']
    tables_to_drop = tables_to_drop.replace(',', ', ')

    sql_query = '''DROP TABLE {}'''.format(
        tables_to_drop
    )

    print(tables_to_drop)
    print(sql_query)

    cursor.execute(sql_query)



    try:
        return jsonify(status_code=200)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...")
        return app.make_response((str(e), 500))





@app.route('/get_raw_table', methods=['POST'])
# @login_required
def get_raw_table():


    # load table content
    form_data = request.form
    sql_query = '''SELECT * FROM {};'''.format(
        form_data['table']
    )


    #result = db_raw.execute(sql_query).fetchall()
    myresult=db_raw.execute(sql_query).fetchall()


    sql_query_two = '''SELECT column_name FROM information_schema.columns WHERE table_name = N'{}' '''.format(
        form_data['table']
    )
    mycols = db_raw.execute(sql_query_two).fetchall()

    colcount = len(mycols)

    cols = []
    for col in mycols:
        cols.append(col[0])
    mytable = pd.DataFrame(myresult, columns=cols)

    myres = mytable.head(1000)

    rowcount = len(mytable.index)


    test_res = list(myres.to_records(index=True))
    my_attributes = list(myres.columns.values)
    my_attributes.insert(0,'index')


    final_res = []
    for row in test_res:
        count = 0
        rowentry = {}
        for i in row:
            a = my_attributes[count]
            rowentry[a] = i
            if i != "*":
                #rowentry[my_attributes[count]] = float(rowentry[my_attributes[count]])
                rowentry[my_attributes[count]] = str(rowentry[my_attributes[count]])
            count +=1
        final_res.append(rowentry)


    
    app.logger.debug('Running query: {}'.format(str(sql_query)))
    try:
        #result = db_raw.execute(sql_query).fetchall()
        # print(result)
        return jsonify(status_code=200,
                       #result=[dict(row) for row in result],

                        result=final_res,
                        attributes=my_attributes,
                        rowcount=rowcount,
                        colcount=colcount

                       #attributes=globals()['table_schemas'][form_data['table']]
                       )
    except Exception as e:
        app.logger.error(traceback.format_exc())
        print(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
        return app.make_response(('', 500))


# def build_single_range_explanation()

# kehan modified for range explanation
@app.route('/user_range_explanation', methods=['POST'])
def user_range_explanation():

    form_data = request.form

    select_range_attrs = form_data['select_range_attrs'].split(',')
    select_range_attrs_data = form_data['select_range_attrs_data'].split(',')
    print(form_data['select_range_attrs'],
          form_data['select_range_attrs_data'])
    cond1_a, cond2_a, res_a = "", "", ""
    cond1_b, cond2_b, res_b = "", "", ""

    if len(form_data['$uq-res-1'].split(' --- ')) == 2:
        cond1_a, res_a = form_data['$uq-res-1'].split(' --- ')[
            0], form_data['$uq-res-1'].split(' --- ')[1]

        cond1_b, res_b = form_data['$uq-res-2'].split(' --- ')[
            0], form_data['$uq-res-2'].split(' --- ')[1]
    else:
        cond1_a, cond2_a, res_a = form_data['$uq-res-1'].split(' --- ')[
            0], form_data['$uq-res-1'].split(' --- ')[1], form_data['$uq-res-1'].split(' --- ')[2]
        cond1_b, cond2_b, res_b = form_data['$uq-res-2'].split(' --- ')[
            0], form_data['$uq-res-2'].split(' --- ')[1], form_data['$uq-res-2'].split(' --- ')[2]

    cond1 = [cond1_a, cond1_b]
    cond2 = [cond2_a, cond2_b]

    if 'avg' in form_data['sql-aggregate'].lower() or 'average' in form_data['sql-aggregate'].lower():

        avg_attr = form_data['sql-aggregate'].split('(')[1].split(')')[0]

        c0_c1 = []
        s0_s1 = []
        Ac_Bc = []
        As_Bs = []
        for i in range(len(cond1)):
            sql_query = "SELECT count(*) as count, sum({}) as sum, ".format(avg_attr)

            agg_case = ""
            for j in range(len(select_range_attrs)):
                agg_case += '{} between {} and {} '.format(
                    select_range_attrs[j], select_range_attrs_data[j*2], select_range_attrs_data[j*2 + 1])
                if j < len(select_range_attrs) - 1:
                    agg_case += ' and '

            sql_query += "sum(case when {} then 1 else 0 end) as c{}, sum(case when {} then {} else 0 end) as s{} FROM {} WHERE {} ; ".format(
                agg_case,
                str(i),
                agg_case,
                avg_attr,
                str(i),
                form_data['sql-from'],
                # form_data['sql-select'].split(', ')[0],
                # cond1[i],
                # form_data['sql-select'].split(', ')[1],
                # cond2[i],
                " {} = '{}' ".format(form_data['sql-select'].split(',')[0], cond1[i]) if cond2_a == "" else " {} = '{}' and {} = '{}'  ".format(
                    form_data['sql-select'].split(',')[0], cond1[i], form_data['sql-select'].split(',')[1], cond2[i]),

            )

            app.logger.debug(
                'Running range explanation query: {}'.format(str(sql_query)))

            try:
                qr = QueryRunner.QueryRunner(sql_query, db_raw)

                qr_result = qr.evaluate_query()

                Ac_Bc.append(float(qr_result[0][0]))
                As_Bs.append(float(qr_result[0][1]))
                c0_c1.append(float(qr_result[0][2]))
                s0_s1.append(float(qr_result[0][3]))
                print(qr_result)

                if qr.error_message is not None:
                    flash(qr_result)
                    return app.make_response(('', 500))
                else:
                    continue
            except Exception as e:
                app.logger.error(traceback.format_exc())
                flash("Something wrong has happened, now redirecting...")
                return app.make_response(('', 500))

        intervention = round((( As_Bs[0] - s0_s1[0] + 0.00001) / ( Ac_Bc[0] - c0_c1[0])) / (
            ( As_Bs[1] - s0_s1[1] + 0.00001) / (Ac_Bc[1] - c0_c1[1])), 2)
        aggravation = round(
            ((s0_s1[0] + 0.00001) / (c0_c1[0])) / ((s0_s1[1] + 0.00001) / (c0_c1[1])), 2)
    else:
        c0_c1 = []
        A_B = []
        for i in range(len(cond1)):
            sql_query = "SELECT count(*) as count, sum(case when "

            for j in range(len(select_range_attrs)):
                sql_query += '{} between {} and {} '.format(
                    select_range_attrs[j], select_range_attrs_data[j*2], select_range_attrs_data[j*2 + 1])
                if j < len(select_range_attrs) - 1:
                    sql_query += ' and '

            sql_query += " then 1 else 0 end) as c{}, {} as agg FROM {} WHERE {} ; ".format(
                str(i),
                form_data['sql-aggregate'],
                form_data['sql-from'],
                # form_data['sql-select'].split(', ')[0],
                # cond1[i],
                # form_data['sql-select'].split(', ')[1],
                # cond2[i],
                " {} = '{}' ".format(form_data['sql-select'].split(',')[0], cond1[i]) if cond2_a == "" else " {} = '{}' and {} = '{}'  ".format(
                    form_data['sql-select'].split(',')[0], cond1[i], form_data['sql-select'].split(',')[1], cond2[i]),

            )

            app.logger.debug(
                'Running range explanation query: {}'.format(str(sql_query)))

            try:
                qr = QueryRunner.QueryRunner(sql_query, db_raw)

                qr_result = qr.evaluate_query()

                A_B.append(qr_result[0][0])
                c0_c1.append(qr_result[0][1])

                print(qr_result)

                if qr.error_message is not None:
                    flash(qr_result)
                    return app.make_response(('', 500))
                else:
                    continue
            except Exception as e:
                app.logger.error(traceback.format_exc())
                flash("Something wrong has happened, now redirecting...")
                return app.make_response(('', 500))

        intervention = round((float(A_B[0]) - (c0_c1[0] + 0.00001)) /
                             (float(A_B[1]) - (c0_c1[1] + 0.00001)), 2)
        aggravation = round(
            (float(c0_c1[0]) + 0.00001) / (float(c0_c1[1]) + 0.00001), 2)

    if c0_c1[0] == 0 and c0_c1[1] == 0:
        aggravation = 'DNE'
    elif c0_c1[0] == 0:
        aggravation = '-inf'
    elif c0_c1[1] == 0:
        aggravation = 'inf'

    app.logger.debug([intervention, aggravation])
    return jsonify(status_code=200,
                   intervention=intervention,
                   # attr=select_exp_1,
                   aggravation=aggravation
                   )


# kehan modified for range explanation
@app.route('/range_explanation', methods=['POST'])
def range_explanation():


    form_data = request.form
    app.logger.debug(form_data)

    cond1_a, cond2_a, res_a = "", "", ""
    cond1_b, cond2_b, res_b = "", "", ""

    if len(form_data['$uq-res-1'].split(' --- ')) == 2:
        cond1_a, res_a = form_data['$uq-res-1'].split(' --- ')[
            0], form_data['$uq-res-1'].split(' --- ')[1]

        cond1_b, res_b = form_data['$uq-res-2'].split(' --- ')[
            0], form_data['$uq-res-2'].split(' --- ')[1]
    else:
        cond1_a, cond2_a, res_a = form_data['$uq-res-1'].split(' --- ')[
            0], form_data['$uq-res-1'].split(' --- ')[1], form_data['$uq-res-1'].split(' --- ')[2]
        cond1_b, cond2_b, res_b = form_data['$uq-res-2'].split(' --- ')[
            0], form_data['$uq-res-2'].split(' --- ')[1], form_data['$uq-res-2'].split(' --- ')[2]

    uq_res_1 = "'" + form_data['$uq-res-1'] + "'"
    uq_res_2 = "'" + form_data['$uq-res-2'] + "'"
    uq_res_1 = uq_res_1.replace(" ", "")
    uq_res_2 = uq_res_2.replace(" ", "")

    cond1 = [cond1_a, cond1_b]
    cond2 = [cond2_a, cond2_b]

    attrs = []
    attr_ranges = []

    ranges = form_data['sql-select-exp'].split(' ∧ ')

    all_exp_attrs = form_data['sql-all-exp'].split(',')

    for r in ranges:
        attrs.append(r.split(' = ')[0])
        # Need Delete
        if 'all' in r:
            attr_ranges.append([])
            continue
        ##
        attr_range = r.split(' = ')[1]
        attr_range = attr_range[1:-1].split(' , ')
        attr_ranges.append(attr_range)

    for attr in all_exp_attrs:
        if attr not in attrs:
            attrs.append(attr)
            attr_ranges.append("DNE")

    where_clause = ' AND '.join(map(lambda x: "{} between {} and {}".format(x.split(' = ')[0], x.split(' = ')[1][1:-1].split(' , ')[0],  x.split(' = ')[1][1:-1].split(' , ')[1]),
                                    form_data['sql-select-exp'].split(' ∧ ')))

    aggregate = form_data['sql-aggregate'].lower()
    if 'avg' in aggregate or 'average' in aggregate:
        aggregate = " avg({}) ".format(aggregate.split('(')[1].split(')')[0])


    # predicates = ""

    # if cond2_a == "":
    #     predicates = " {} = '{}' ".format(form_data['sql-select'].split(',')[0],cond1[i])
    # else:
    #     predicates = " {} = '{}' and {} = '{}' ' ".format(form_data['sql-select'].split(',')[0],cond1[i])

    # select_exp_1 = form_data['sql-select-exp'].split(' ∧ ')[0].split(' = ')[0]
    # select_exp_2 = form_data['sql-select-exp'].split(' ∧ ')[1].split(' = ')[0]
    query_results = []
    for j, attr in enumerate(attrs):
        attr_result = []
        for i in range(len(cond1)):
            sql_query = '''SELECT {}, {} as {}, {} as group FROM {} WHERE {}  GROUP BY {} ;'''.format(
                attr,
                aggregate,
                '"' + aggregate + '"',
                uq_res_1 if i == 0 else uq_res_2,
                # str(i),
                form_data['sql-from'],
                # form_data['sql-select'].split(',')[0],
                # cond1[i],
                # form_data['sql-select'].split(',')[1],
                # cond2[i],
                " {} = '{}' ".format(form_data['sql-select'].split(',')[0], cond1[i]) if cond2_a == "" else " {} = '{}' and {} = '{}'  ".format(
                    form_data['sql-select'].split(',')[0], cond1[i], form_data['sql-select'].split(',')[1], cond2[i]),
                attr
            )
            app.logger.debug(
                'Running range explanation query: {}'.format(str(sql_query)))
            try:
                qr = QueryRunner.QueryRunner(sql_query, db_raw)
                qr_result = qr.evaluate_query()
                qr_result = list(qr_result)
                attr_result += qr_result
                if qr.error_message is not None:
                    flash(qr_result)
                    return app.make_response(('', 500))
                else:
                    continue
            except Exception as e:
                app.logger.error(traceback.format_exc())
                flash("Something wrong has happened, now redirecting...")
                return app.make_response(('', 500))
        query_results.append([attr, [dict(row) for row in attr_result]])

    interv_change_results = []

    for j, attr in enumerate(attrs):
        attr_result = []
        for i in range(len(cond1)):
            sql_query = '''SELECT {}, {} as {}, {} as group FROM {} WHERE {} and {}({})  GROUP BY {} ;'''.format(
                attr,
                aggregate,
                '"' + aggregate + '"',
                uq_res_1 if i == 0 else uq_res_2,
                # str(i),
                form_data['sql-from'],
                # form_data['sql-select'].split(',')[0],
                # cond1[i],
                # form_data['sql-select'].split(',')[1],
                # cond2[i],
                " {} = '{}' ".format(form_data['sql-select'].split(',')[0], cond1[i]) if cond2_a == "" else " {} = '{}' and {} = '{}'  ".format(
                    form_data['sql-select'].split(',')[0], cond1[i], form_data['sql-select'].split(',')[1], cond2[i]),
                ' NOT ',
                where_clause,
                attr
            )
            app.logger.debug(
                'Running range attr explanation query: {}'.format(str(sql_query)))
            try:
                qr = QueryRunner.QueryRunner(sql_query, db_raw)
                qr_result = qr.evaluate_query()
                qr_result = list(qr_result)
                attr_result += qr_result
                if qr.error_message is not None:
                    flash(qr_result)
                    return app.make_response(('', 500))
                else:
                    continue
            except Exception as e:
                app.logger.error(traceback.format_exc())
                flash("Something wrong has happened, now redirecting...")
                return app.make_response(('', 500))
        interv_change_results.append(
            [attr, [dict(row) for row in attr_result]])

    aggr_change_results = []

    for j, attr in enumerate(attrs):
        attr_result = []
        for i in range(len(cond1)):
            sql_query = '''SELECT {}, {} as {}, {} as group FROM {} WHERE {} and {}({})  GROUP BY {} ;'''.format(
                attr,
                aggregate,
                '"' + aggregate + '"',
                uq_res_1 if i == 0 else uq_res_2,
                # str(i),
                form_data['sql-from'],
                # form_data['sql-select'].split(',')[0],
                # cond1[i],
                # form_data['sql-select'].split(',')[1],
                # cond2[i],
                " {} = '{}' ".format(form_data['sql-select'].split(',')[0], cond1[i]) if cond2_a == "" else " {} = '{}' and {} = '{}'  ".format(
                    form_data['sql-select'].split(',')[0], cond1[i], form_data['sql-select'].split(',')[1], cond2[i]),
                '',
                where_clause,
                attr
            )
            app.logger.debug(
                'Running range attr explanation query: {}'.format(str(sql_query)))
            try:
                qr = QueryRunner.QueryRunner(sql_query, db_raw)
                qr_result = qr.evaluate_query()
                qr_result = list(qr_result)
                attr_result += qr_result
                if qr.error_message is not None:
                    flash(qr_result)
                    return app.make_response(('', 500))
                else:
                    continue
            except Exception as e:
                app.logger.error(traceback.format_exc())
                flash("Something wrong has happened, now redirecting...")
                return app.make_response(('', 500))
        aggr_change_results.append([attr, [dict(row) for row in attr_result]])

    return jsonify(status_code=200,
                   query_results=query_results,
                   interv_change_results=interv_change_results,
                   aggr_change_results=aggr_change_results,
                   attr_ranges=attr_ranges,
                   aggr=aggregate
                   )


@app.route('/run_interv_query', methods=['POST'])
# @login_required
def run_interv_query():

    form_data = request.form
    app.logger.debug(form_data)


    globals()['db_raw'] = db.get_engine(app, bind='raw')
    cursor = db_raw
    sql_query = '''SELECT * FROM {} limit 1000;'''.format(
        form_data['sql-from']
    )
    myresult = cursor.execute(sql_query).fetchall()

    sql_query_two = '''SELECT column_name FROM information_schema.columns WHERE table_name = N'{}' '''.format(
        form_data['sql-from']
    )
    mycols = cursor.execute(sql_query_two).fetchall()


    cols = []
    for col in mycols:
        cols.append(col[0])
    df = pd.DataFrame(myresult, columns=cols)


    treat_var = request.form.get('xvar')
    out_var = request.form.get('y-var')

    model = dame_flame.matching.FLAME()
    model.fit(df, treatment_column_name=treat_var, outcome_column_name=out_var)
    myres = model.predict(df)
    result = myres.to_dict()

    engine = db.get_engine(bind='raw')


    myres.to_sql(name="res_table", con=engine, if_exists='replace', index=False)

    #test_query = '''SELECT * FROM res_table'''
    #test_result = cursor.execute(test_query).fetchall()

    #test_df = pd.DataFrame(test_result)
    #print(test_df)

    #print(myres)

    ate = str(dame_flame.utils.post_processing.ATE(matching_object=model))
    att = str(dame_flame.utils.post_processing.ATT(matching_object=model))









    # build query from input form
    sql_query = '''SELECT {}, {} FROM {}{} GROUP BY {} ;'''.format(
        form_data['sql-select'],
        form_data['sql-aggregate'],
        form_data['sql-from'],
        ' WHERE ' + form_data['sql-add-where'] if 'sql-add-where' in form_data
                                                  and len(form_data['sql-add-where'].strip()) > 0 else '',
        form_data['sql-select']
    )
    app.logger.debug('Running query: {}'.format(str(sql_query)))

    # sql-where is the explanation predicate selected by user
    # Kehan modified for range explanation
    if form_data['range-exp'] == 'False':
        where_clause = ' AND '.join(map(lambda x: "{}='{}'".format(x.split(' = ')[0], x.split(' = ')[1]),
                                        form_data['sql-where'].split(' ∧ ')))
    else:
        where_clause = ' AND '.join(map(lambda x: "{} between {} and {}".format(x.split(' = ')[0], x.split(' = ')[1][1:-1].split(' , ')[0],  x.split(' = ')[1][1:-1].split(' , ')[1]),
                                        form_data['sql-select-exp'].split(' ∧ ')))
        print(where_clause)

    # build interv/aggrav query from user question and user selected explanation
    sql_interv_query = '''SELECT {}, {} FROM {} WHERE {}{}({}) GROUP BY {};'''.format(
        form_data['sql-select'],
        form_data['sql-aggregate'],
        form_data['sql-from'],
        (form_data['sql-add-where'] + ' AND ') if 'sql-add-where' in form_data
                                                  and len(form_data['sql-add-where'].strip()) > 0 else '',
        ' NOT ' if form_data['is-interv'] == 'true' else '',
        where_clause,
        form_data['sql-select']
    )
    app.logger.debug('Running interv query: {}'.format(str(sql_interv_query)))
    groupby_attributes = list(
        map(lambda x: x.strip(), form_data['sql-select'].split(',')))

    # return both raw result and the interv/aggrav result
    try:
        qr = QueryRunner.QueryRunner(sql_query, db_raw)
        qr_result = list(map(
            lambda x: dict(zip(groupby_attributes + [form_data['sql-aggregate']],
                               list(map(lambda y: y.strip() if isinstance(y, str) else y,
                                        x)))),
            qr.evaluate_query()))
        qr.sql_query = sql_interv_query
        qr_interv_result = list(map(
            lambda x: dict(zip(groupby_attributes + [form_data['sql-aggregate']],
                               list(map(lambda y: y.strip() if isinstance(y, str) else y,
                                        x)))),
            qr.evaluate_query()))
        if qr.error_message is not None:
            flash(qr_result)
            return app.make_response(('', 500))
        else:
            return jsonify(status_code=200,
                           query_result=qr_result,
                           interv_query_result=qr_interv_result,
                           groupby_attributes=groupby_attributes,
                           aggregation=form_data['sql-aggregate'],
                           ate=ate,
                           att=att,
                           #result=[dict(row) for row in myres.iterrows()],
                           result=result,
                           attributes=globals()['table_schemas']['res_table'])
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...")
        return app.make_response(('', 500))

    # return jsonify(status_code=200, ra_result=rr.get_evaluate_result())


@app.route('/explain_query_result', methods=['POST'])
# @login_required
def explain_query_result():

    form_data = request.form
    app.logger.debug(form_data)
    # build query from input form

    sql_query = '''SELECT {}, {} FROM {}{} GROUP BY {} ;'''.format(
        form_data['sql-select'],
        form_data['sql-aggregate'],
        form_data['sql-from'],
        ' WHERE ' + form_data['sql-add-where'] if 'sql-add-where' in form_data
                                                  and len(form_data['sql-add-where'].strip()) > 0 else '',
        form_data['sql-select']
    )
    groupby_attributes = list(
        map(lambda x: x.strip(), form_data['sql-select'].split(',')))

    app.logger.debug('Running query: {}'.format(str(sql_query)))
    try:
        attr_list = list(table_schemas[form_data['sql-from']])
        app.logger.error(attr_list)
        selected_attr = []
        for attr in attr_list:
            if not (attr not in form_data or form_data[attr] == False or str(form_data[attr]).strip() == 'false'):
                selected_attr.append(attr)

        # hardcode some blacklisted attributes
        # attr_list.remove('capital_gain')
        # attr_list.remove('capital_loss')
        # attr_list.remove('hours_per_week')
        # attr_list.remove('education_num')
        # attr_list.remove('age')
        # attr_list.remove('fnlwgt')
        # attr_list.remove('relationship')

        attr_list = [x for x in attr_list if x not in groupby_attributes]
        selected_attr = [
            x for x in selected_attr if x not in groupby_attributes]
        # if form_data['range-exp'] == 'True':
        # app.logger.debug(globals()['table_numeric_attr'][form_data['sql-from']])

        # print("=======range-exp========", form_data['range-exp'])

        # use attributes selected by user to generate explanations
        intf = IntervFinder.IntervFinder(sql_query, db_raw, selected_attr,  # attr_list,
                                         form_data['sql-from'], groupby_attributes,
                                         form_data['sql-aggregate'],
                                         form_data['uq-res-1'], form_data['uq-res-2'],
                                         form_data['uq-direction'], form_data['uq-topk'],
                                         form_data['uq-p-pred'],
                                         json.loads(
                                             form_data['predicate-blacklist']),
                                         form_data['range-exp'],
                                         form_data['uq-min-dp'],
                                         form_data['uq-max-dp']
                                         )

        interv_expls = intf.find_explanation(True)
        aggrav_expls = intf.find_explanation(False)
        # app.logger.debug(interv_expls)
        result_schema = selected_attr + ['score']
        interv_res = list(map(
            lambda x: list(filter(lambda z: not (z.endswith('-9999') or z.endswith('-9999.0') or z.endswith('?')),
                                  list(map(lambda y: "{} = {}".format(result_schema[y[0]], str(y[1]).strip()),
                                           enumerate(x))))),
            interv_expls))
        aggrav_res = list(map(
            lambda x: list(filter(lambda z: not (z.endswith('-9999') or z.endswith('-9999.0') or z.endswith('?')),
                                  list(map(lambda y: "{} = {}".format(result_schema[y[0]], str(y[1]).strip()),
                                           enumerate(x))))),
            aggrav_expls))

        interv_res = list(filter(lambda x: len(x) > 1, interv_res))[
            :int(form_data['uq-topk'])]
        aggrav_res = list(filter(lambda x: len(x) > 1, aggrav_res))[
            :int(form_data['uq-topk'])]

        app.logger.debug(interv_res)
        # app.logger.debug(aggrav_res)

        def cluster_exp(exp_list, interv):
            exp_list = [[pred for pred in exp if 'all' not in pred]
                        for exp in exp_list]

            predicates = list(
                set([pred for exp in exp_list for pred in exp if 'score = ' not in pred]))

            clustered_list = {}
            best_score = {}

            for pred in predicates:
                clustered_list[pred] = []
                for exp in exp_list:
                    if pred in exp:
                        exp_to_add = exp.copy()
                        exp_to_add.remove(pred)
                        exp_to_add.insert(0, pred)
                        clustered_list[pred].append(exp_to_add)
                        if pred not in best_score:
                            best_score[pred] = exp[-1].split(" = ")[1]
                        else:
                            if interv and float(best_score[pred]) > float(exp[-1].split(" = ")[1]):
                                best_score[pred] = exp[-1].split(" = ")[1]
                            if not interv and float(best_score[pred]) < float(exp[-1].split(" = ")[1]):
                                best_score[pred] = exp[-1].split(" = ")[1]
                        print(pred, best_score[pred], exp[-1].split(" = ")[1])

            if interv:
                sorted_clustered_list = {k: v for k, v in sorted(
                    clustered_list.items(), key=lambda item: float(best_score[item[0]]))}
            else:
                sorted_clustered_list = {k: v for k, v in sorted(
                    clustered_list.items(), key=lambda item: -1 * float(best_score[item[0]]))}

            # app.logger.debug(best_score)
            # app.logger.debug(sorted_clustered_list)

            counter = 1
            expand_control = []
            on_top = True
            for k in sorted_clustered_list:
                bs = 'bestscore = '+str(best_score[k])
                top = None
                for exp in sorted_clustered_list[k]:
                    if on_top:
                        expand_control.append(counter)
                        on_top = False
                    else:
                        expand_control.append(counter*-1)
                    exp.append(bs)
                    if len(exp) == 3:
                        top = exp
                if top != None:
                    sorted_clustered_list[k].remove(top)
                    sorted_clustered_list[k].insert(0, top)
                else:
                    sorted_clustered_list[k].insert(0, [k, 'score = N/A', bs])
                    if on_top:
                        expand_control.append(counter)
                        on_top = False
                    else:
                        expand_control.append(counter*-1)

                counter += 1
                on_top = True

            flattened_clustered_list = [
                exp for k in sorted_clustered_list for exp in sorted_clustered_list[k]]
            # print(expand_control)

            # app.logger.debug(flattened_clustered_list)

            return flattened_clustered_list, expand_control

        interv_res, interv_expand_control = cluster_exp(interv_res, True)
        aggrav_res, aggrav_expand_control = cluster_exp(aggrav_res, False)

        return jsonify(status_code=200,
                       explanations=[interv_res, aggrav_res],
                       groupby_attributes=groupby_attributes,
                       expand_controls=[interv_expand_control,
                                        aggrav_expand_control],
                       aggregation=form_data['sql-aggregate'])
    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...")
        return app.make_response(('', 500))


@app.route('/recommend_drop', methods=['POST'])
#@login_required
def recommend_drop(): 

    print('test 1')

    form_data = request.form

    print("test 2")

    globals()['db_raw'] = db.get_engine(app, bind='raw')
    cursor = db_raw
    sql_query = '''SELECT * FROM {}'''.format(
        form_data['sql-from']
    )
    myresult = cursor.execute(sql_query).fetchall()

    sql_query_two = '''SELECT column_name FROM information_schema.columns WHERE table_name = N'{}' '''.format(
        form_data['sql-from']
    )
    mycols = cursor.execute(sql_query_two).fetchall()


    cols = []
    for col in mycols:
        cols.append(col[0])
    df = pd.DataFrame(myresult, columns=cols)

    x_recc = form_data['x-recc']
    y_recc = form_data['y-recc']


    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    model = dame_flame.matching.FLAME(verbose=3)
    model.fit(df, treatment_column_name=x_recc, outcome_column_name=y_recc)
    result=model.predict(df)

    output = new_stdout.getvalue()
    sys.stdout = old_stdout

    print(result)

    out_list = output.split("Iteration number: ")
    out_list_two = []

    i = 0
    for iternum in out_list: 
        entry = iternum.split("\n")
        out_list_two.append(entry)
        i+=1

    out_list_two = out_list_two[2:]

    dropped_ordered = []
    for elem in out_list_two: 
        dropped_ordered.append(elem[7].split(' ')[8])

    num_recs = int(form_data['numrecs'])
    print("numrecs = " + str(num_recs))

    num_recs_dict = []
    i = 1
    if num_recs >= len(dropped_ordered):
        #d = {'rank': range(start=1, stop=len(dropped_ordered)+1), 'var': dropped_ordered}
        for item in dropped_ordered: 
            d = {'rank': i, 'var': item}
            num_recs_dict.append(d)
            i +=1
    else:
        #d = {'rank': range(start=1, stop=num_recs+1), 'var': dropped_ordered[:num_recs]}
        while i <=num_recs: 
            d = {'rank': i, 'var': dropped_ordered[i-1]}
            num_recs_dict.append(d)
            i +=1
    attributes = ['rank', 'var']

    print(num_recs_dict)
    
    return jsonify(status_code=200,
                   recommended_vars=num_recs_dict,
                   attributes=attributes)






@app.route('/recommend_attributes', methods=['POST'])
# @login_required
def recommend_attributes():



    form_data = request.form
    # build query from input form
    app.logger.debug(form_data)
    sql_query = '''SELECT {}, {} FROM {}{} GROUP BY {} ;'''.format(
        form_data['sql-select'],
        form_data['sql-aggregate'],
        form_data['sql-from'],
        ' WHERE ' + form_data['sql-add-where'] if 'sql-add-where' in form_data
                                                  and len(form_data['sql-add-where'].strip()) > 0 else '',
        form_data['sql-select']
    )
    groupby_attributes = list(
        map(lambda x: x.strip(), form_data['sql-select'].split(',')))

    app.logger.debug('Running query: {}'.format(str(sql_query)))

# try:
    attr_list = list(table_schemas[form_data['sql-from']])
    # selected_attr = []
    # for attr in attr_list:
    #     if not (attr not in form_data or form_data[attr] == False or str(form_data[attr]).strip() == 'false'):
    #         selected_attr.append(attr)

    # hardcode some blacklisted attributes
    # attr_list.remove('capital_gain')
    # attr_list.remove('capital_loss')
    # attr_list.remove('hours_per_week')
    # attr_list.remove('education_num')
    # attr_list.remove('age')
    # attr_list.remove('fnlwgt')
    # attr_list.remove('education_num_group')
    attr_list = [x for x in attr_list if x not in groupby_attributes]
    disabled_list = []
    # selected_attr = [x for x in selected_attr if x not in groupby_attributes]

    # kehan modified for range exp
    # app.logger.debug(form_data['range-exp'], globals()['table_numeric_attr'][form_data['sql-from']])
    if form_data['range-exp'] == 'true':
        cont_attrs = [x[0] for x in globals()['table_numeric_attr']
                      [form_data['sql-from']]]
        disabled_list = [x for x in attr_list if not x in cont_attrs]
        attr_list = [x for x in attr_list if x in cont_attrs]
        # app.logger.debug(cont_attrs, attr_list)

    rec_attr_list = attr_rmd.recommend_attributes(sql_query, attr_list,
                                                  form_data['sql-select'],
                                                  form_data['sql-from'], groupby_attributes,
                                                  form_data['uq-res-1'], form_data['uq-res-2'],
                                                  form_data['uq-direction'], [], int(form_data['rec-k-attr']))

    return jsonify(status_code=200,
                   recommended_attributes=rec_attr_list,
                   groupby_attributes=groupby_attributes,
                   disabled_list=disabled_list)
    # except Exception as e:
    #     # app.logger.error(traceback.format_exc())
    #     flash("Something wrong has happened, now redirecting...")
    #     return app.make_response(('', 500))


@app.route('/reset_recommender_weights', methods=['POST'])
def reset_recommender_weights():
    attr_rmd.weights[0] = 0.7
    attr_rmd.weights[1] = 0.3
    return jsonify(status_code=200)


@app.route('/user_feedback', methods=['POST'])
# @login_required
def user_feedback():

    form_data = request.form
    predicate_list = json.loads(form_data['predicate-list'])
    user_score = float(form_data['user-score'])
    app.logger.debug(user_score)

    try:
        as_score = attr_rmd.last_score1
        rf_score = attr_rmd.last_score2

        sum0 = 0
        sum1 = 0
        for k in as_score:
            sum0 += as_score[k]
        for k in rf_score:
            sum1 += rf_score[k]

        app.logger.debug(sum0)
        app.logger.debug(sum1)

        app.logger.debug('Scores of ASM and RF:')
        app.logger.debug(str(as_score) + ', ' + str(rf_score))
        app.logger.debug(str(user_score) + ', ' + str(attr_rmd.weights))
        if attr_rmd.last_one_var:
            return app.make_response(('', 200))
        else:
            for a in predicate_list:
                arr = a.split(' = ')
                if arr[0] not in as_score or arr[0] not in rf_score:
                    continue

                # kehan modified for range explanation and '-9999.0'
                if arr[1] != 'all' or arr[1] == '-9999.0':
                    print(arr)
                    diff = as_score[arr[0]] - rf_score[arr[0]]
                    attr_rmd.weights[0] += diff * user_score * \
                        AttributeRecommender.AttributeRecommender.adjust_rate
                    attr_rmd.weights[1] -= diff * user_score * \
                        AttributeRecommender.AttributeRecommender.adjust_rate

            weights_sum = attr_rmd.weights[0] + attr_rmd.weights[1]
            attr_rmd.weights[0] /= weights_sum
            attr_rmd.weights[1] /= weights_sum
            print(weights_sum, attr_rmd.weights[0], attr_rmd.weights[1])

            new_recom_scores = []
            # select_attr = form_data['sql-select'].split()

            print(as_score, rf_score)

            for attr, tp in table_datatype[form_data['sql-from']]:
                try:
                    new_recom_scores.append(
                        [attr, attr_rmd.weights[0] * as_score[attr] + attr_rmd.weights[1] * rf_score[attr]])
                except:
                    continue

            print(new_recom_scores)
            return jsonify(status_code=200,
                           new_recom_scores=new_recom_scores)

    except Exception as e:
        app.logger.error(traceback.format_exc())
        flash("Something wrong has happened, now redirecting...\nIf it continues, please contact course staff.")
        return app.make_response(('', 500))


@app.route("/upload_table", methods=['POST'])
def upload_table():

    target = os.path.join(APP_ROOT, "DataSource/")

    if not os.path.isdir(target):
        os.mkdir(target)

    if 'file' not in request.files:
        error = "Missing data source!"
        return jsonify({'error': error})

    file = request.files['file']
    #     with open(file) as csvfile:
    # ut_reader = csv.reader(csvfile, delimiter=',')
    table_name = file.filename.split('.')[0]
    file_type = file.filename.split('.')[1]
    engine = db.get_engine(bind='raw')


    if file_type=='csv': 
        df = pd.read_csv(file, skipinitialspace=True)
    elif file_type=='xlsx' or file_type=='xls': 
        df = pd.read_excel(file)
    #except: 
    #    raise Exception("please upload .csv file or excel file")



    # kehan done
    #df = pd.read_csv(file, skipinitialspace=True)

    my_size = len(df.index)







    df.columns = df.columns.str.replace('-', '_')
    df.columns = df.columns.str.replace(' ', '')
    df.columns = map(str.lower, df.columns)

    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.dropna()
    print(df)

    # string_cols = list(df.select_dtypes(include=['str']).columns.values)

    # for str_col in string_cols:
    #     df[str_col] = df[str_col].str.lower()

    df.head(0).to_sql(table_name, engine, if_exists='replace', index=False)
    # print(":"+df['workclass'].iloc[0]+":")

    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    # contents = output.getvalue()
    cur.copy_from(output, table_name, null="")  # null values become ''
    conn.commit()
    success = "Success!"
    return jsonify({'file': success, 'my_size': my_size, 'table_name': table_name})


def init_db():
    # db.init_app(app)
    # db.app = app
    # db.create_all()
    # db.init_app(app)
    # with current_app.app_context():
    with app.app_context():
        db.init_app(app)
        # db.app = current_app
        db.create_all()
        db.session.commit()
        globals()['db_raw'] = db.get_engine(app, bind='raw')
        cursor = db_raw
        table_query = '''
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_schema, table_name;
        '''
        globals()['available_tables'] = list(map(
            lambda x: x[0],
            cursor.execute(table_query).fetchall()
        ))
        if 'ra_users' in globals()['available_tables']:
            globals()['available_tables'].remove('ra_users')

        # globals()['available_tables'] = ['adult']
        globals()['table_schemas'] = {}
        globals()['table_datatype'] = {}
        globals()['table_numeric_attr'] = {}
        for tbl in globals()['available_tables']:
            globals()['table_schemas'][tbl] = cursor.execute(
                "select * FROM {} LIMIT 1;".format(tbl)).keys()
            globals()['table_datatype'][tbl] = cursor.execute('''
                select column_name, data_type 
                from information_schema.columns 
                where table_name = '{}';
                '''.format(tbl)).fetchall()
            globals()['table_numeric_attr'][tbl] = cursor.execute('''
                select column_name
                from information_schema.columns 
                where table_name = '{}'
                and data_type in ('smallint', 'integer', 'bigint', 
                            'decimal', 'numeric', 'real', 'double precision',
                            'smallserial', 'serial', 'bigserial', 'money');
                '''.format(tbl)).fetchall()


init_db()
attr_rmd = AttributeRecommender.AttributeRecommender(db_raw)

if __name__ == '__main__':
    # app = create_app()

    init_db()
    app.run()
