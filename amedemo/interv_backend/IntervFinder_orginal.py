import sys
import re
import psycopg2
import datetime
import logging
import pandas as pd


def create_tables_helper(is_interv, table, groupby_attr, queries, var, equation, dir, num_queries, sample_rate, num_pred):

    query = ""
    gb = groupby_attr

    use_sample = False
    if sample_rate < 100.0:
        query += " sample as (select * from " + table + " tablesample system (" + str(sample_rate) + ") ) , "
        use_sample = True

    # predicate of user question tuples
    p0 = queries[0].split(" --- ")[:-1]
    p1 = ["", ""]

    if num_queries == 2:
        p1 = queries[1].split(" --- ")[:-1]

    for a in range(num_queries):
        query += "t" + str(a) + " as ( select "
        for i in range(len(var)):
            if var[i] in ['video_release', 'rated_at', 'release_date']:
                continue
            query += "coalesce(" + var[i] + "," + "'-9999') as v" + str(i) + ","

        query += " count(*) as c{}, ({}) as pc{} from {}".format(
            str(a),
            " + ".join(map(lambda x: "case when {} is null then 0 else 1 end".format(x), var)),
            str(a),
            "sample" if use_sample else table
        )

        if a == 0:
            query += " where {}".format(" and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p0))))
        elif a == 1:
            query += " where {}".format(" and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p1))))

        query += " group by cube("
        sj = ", ".join(var)
        query += sj + ")),"

    if use_sample:
        query += "A as ( select count(*) from sample where {}),".format(
            " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p0))))
        if num_queries == 2:
            query += "B as ( select count(*) from sample where {}),".format(
                " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p1))))
    else:
        query += "A as ( select count(*) from {} where {}),".format(
            table,
            " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p0)))
        )
        if num_queries == 2:
            query += "B as ( select count(*) from {} where {}),".format(
                table,
                " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p1)))
            )

    query += "intervention as (select "
    for a in range(num_queries):
        for i in range(len(var)):
            query += "t" + str(a) + ".v" + str(i) + " as " + "t" + str(a) + "v" + str(i) + ","

    if num_queries == 2:
        if is_interv:
            query += " A.count - coalesce(t0.c0,0)+0.00001 as c0n, B.count- coalesce(t1.c1,0)+0.00001 as c1n from A,B, t0 left outer join t1 on "
        else:
            query += " coalesce(t0.c0,0)+0.00001 as c0n, coalesce(t1.c1,0)+0.00001 as c1n from A,B, t0 left outer join t1 on "
        for a in range(num_queries-1):
            for i in range(len(var)):
                query += "t" + str(a) + ".v" + str(i) + " = " + "t" + str((a+1)) + ".v" + str(i) + " and "
        query = query[:-4]
        query += " where t0.pc0 <= {} and t1.pc1 <= {}".format(str(num_pred), str(num_pred))
    else:
        if is_interv:
            query += " A.count - coalesce(t0.c0,0)+0.00001 as c0n from A, t0"
        else:
            query += " coalesce(t0.c0,0)+0.00001 as c0n from A, t0"
        query += " where t0.pc0 <= {}".format(str(num_pred))

    query += ") "
    query += "select distinct "

    for i in range(len(var)):
        if num_queries == 2:
            query += "coalesce("

        for a in range(num_queries):
            query += "intervention." + "t" + str(a) + "v" + str(i) + ","
        query = query[:-1]
        if num_queries == 2:
            query += ")"
        query += " as t" + str(i) + ","

    query += equation + " from intervention order by evalue "
    if dir == 'high':
        if is_interv:
            query += "asc"
        else:
            query += "desc"
    else:
        if is_interv:
            query += "desc"
        else:
            query += "asc"

    return query


class IntervFinder:

    def __init__(self, sql_query, db_raw, attrs, table, groupby_attr, uq1, uq2, uq_dir, topk=10, ppred=3, predicate_blacklist=None):
        self.sql_query = sql_query
        self.attrs = attrs
        self.error_message = None
        self.cur = db_raw
        self.table = table
        self.groupby_attr = groupby_attr
        self.uq1 = uq1
        self.uq2 = uq2
        self.dir = uq_dir
        self.topk = int(topk)
        self.ppred = int(ppred)
        self.predicate_blacklist = predicate_blacklist

    def find_explanation(self, interv=True):

        try:
            if self.uq2 == "None":
                if interv:
                    query = self.build_interv_query(self.attrs, self.uq1, None, None, self.dir,self.table, self.groupby_attr, 100)
                else:
                    query = self.build_aggrav_query(self.attrs, self.uq1, None, None, self.dir, self.table,
                                                    self.groupby_attr, 100)
            else:
                if interv:
                    query = self.build_interv_query(self.attrs, self.uq1, self.uq2, '/', self.dir,self.table, self.groupby_attr, 100)
                else:
                    query = self.build_aggrav_query(self.attrs, self.uq1, self.uq2, '/', self.dir, self.table,
                                                    self.groupby_attr, 100)
            print(query)

            blacklist = []
            if self.predicate_blacklist is not None:
                for i in range(len(self.predicate_blacklist)):
                    blacklist.append([])
                    for j in range(len(self.predicate_blacklist[i])):
                        arr = self.predicate_blacklist[i][j].split(' = ')
                        for k in range(len(self.attrs)):
                            if self.attrs[k] == arr[0]:
                                blacklist[-1].append("t{} = '{}'".format(str(k), arr[1]))
                                break

            logging.debug(blacklist)
            if len(blacklist) > 0:
                input = '''
                select * from  ({}) t where not ({}) limit {};
                '''.format(query,
                           " OR ".join(map(lambda y: "(" + " AND ".join(y) + ")", blacklist)),
                           str(self.topk * 2))
            else:
                input = '''select * from  ({}) t limit {};'''.format(query, str(self.topk * 2))
            logging.debug('Running explanation query: {}'.format(input))
            result = self.cur.execute(input).fetchall()
            print("====result=====")
            print(result[:10])
            print("====result=====")
            return result

        except psycopg2.Error as e:
            self.error_message = 'There is something wrong with your query!'
            return e

    def recommend_attrs(self, attrs, q1, q2, table, groupby_attr):
        q1_value = q1.split(' --- ')
        where_clause1 = ' AND '.join(list(map(lambda y: "({}='{}')".format(y[1], q1_value[y[0]]), enumerate(groupby_attr))))
        if q2 != 'None' and q2 is not None:
            q2_value = q2.split(' --- ')
            where_clause2 = ' AND '.join(list(map(lambda y: "({}='{}')".format(y[1], q2_value[y[0]]), enumerate(groupby_attr))))
            sql_cmd = '''SELECT * FROM {} WHERE ({}) OR ({})'''.format(table, where_clause1, where_clause2)
        else:
            sql_cmd = '''SELECT * FROM {} WHERE {}'''.format(table, where_clause1)
        df = pd.read_sql(sql_cmd, self.cur)
        return (df)

    def build_interv_query(self, attrs, q1, q2, op, dir, table, groupby_attr, sample_rate):
        print(attrs, q1, q2, op, dir, table, groupby_attr, sample_rate)

        queries = [None, None]
        var = attrs
        if q2 == 'None' or q2 is None:
            queries[0] = q1
            equation = "cast(c0n AS float) as Evalue"
            num_queries = 1
        else:
            queries[0] = q1
            queries[1] = q2
            equation = "cast(c0n AS float) " + op + " cast(c1n as float) as EValue";
            num_queries = 2

        interv_query = "with "
        interv_query += create_tables_helper(True, table, groupby_attr, queries,
                                             var, equation, dir, num_queries, sample_rate, self.ppred)
        print(interv_query)
        return interv_query

    def build_aggrav_query(self, attrs, q1, q2, op, dir, table, groupby_attr, sample_rate):

        queries = [None, None]
        var = attrs
        if q2 == 'None' or q2 is None:
            queries[0] = q1
            equation = "cast(c0n AS float) as Evalue"
            num_queries = 1
        else:
            queries[0] = q1
            queries[1] = q2
            equation = "cast(c0n AS float) " + op + " cast(c1n as float) as EValue";
            num_queries = 2

        aggrav_query = "with "

        aggrav_query += create_tables_helper(False, table, groupby_attr, queries,
                                             var, equation, dir, num_queries, sample_rate, self.ppred)

        return aggrav_query

