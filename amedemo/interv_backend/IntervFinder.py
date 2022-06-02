import sys
import re
import psycopg2
import datetime
import logging
import pandas as pd

# Kehan modified for continuous attr
def create_tables_helper(is_interv, table, groupby_attr, aggregate, queries, var, equation, dir, num_queries, sample_rate, num_pred, range_exp, sel_min, sel_max):


    print("=======aggregate==========", aggregate)
    print("create_tables_helper range exp:", range_exp)
    print("sel:", sel_min, sel_max)
    num_of_bucket = 15.0
    sel_min /= 100
    sel_max /= 100
    continuous_attr = True

    is_avg_agg = False

    aggregate = aggregate.lower()
    if "avg" in aggregate or "average" in  aggregate:
        is_avg_agg = True



    if str(range_exp) == 'true':
        continuous_attr = True
    else:
        continuous_attr = False

    if continuous_attr:

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


        for i in range(len(var)):
            query += 'v{}_stat as (select min({}) as v{}_min, max({}) as v{}_max  from {} ),'.format(str(i), var[i],str(i), var[i], str(i), table)


        for a in range(num_queries):
            query += "t" + str(a) + " as ( select "
            for i in range(len(var)):
                query += "coalesce( width_bucket({} , cast(v{}_min as numeric) , cast( v{}_max as numeric), cast({} as int )), '-9999') as bucket_v{}, ".format(var[i],str(i),str(i), str(num_of_bucket) ,str(i))
            
            query += '{} as agg, count(*) as c{}  from {}, '.format(aggregate, str(a), table)

            for i in range(len(var)):
                query += 'v{}_stat'.format(str(i))
                if i < len(var) -1:
                    query += ","


            if a == 0:
                query += " where {} ".format(" and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p0))))

            elif a == 1:
                query += " where {} ".format(" and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p1))))

            query += " group by cube("
            for i in range(len(var)):
                query += "width_bucket({} , cast(v{}_min as numeric) , cast( v{}_max as numeric), cast({} as int )) ".format(var[i],str(i),str(i), str(num_of_bucket) )
                if i < len(var) -1:
                    query += ","      
            query += ")),"


   
        for a in range(num_queries):
            if a == 0:
                query += " A " + " as ( select "
            elif a == 1:
                query += " B " + " as ( select "
            query += "{} as agg, COUNT(*) as count from {}".format(aggregate, table)

            if a == 0:
                query += " where {}".format(" and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p0))))


            elif a == 1:
                query += " where {} ".format(" and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p1))))
            query += '),'


        query += "intervention as (select "
        for i in range(len(var)):
            query += "t0.bucket_v{} as bv{},".format(str(i), str(i))


        print(is_interv)
        if num_queries == 2:
            if is_interv:
                if is_avg_agg:
                    query += " (A.agg * A.count - t0.agg * t0.c0 ) / ((A.count - coalesce(t0.c0,0))+0.00001) as c0n, (B.agg * B.count - t1.agg * t1.c1 ) / ((B.count - coalesce(t1.c1,0))+0.00001)  as c1n from t0 "
                else:
                    query += " ABS(A.count - coalesce(t0.c0,0))+0.00001 as c0n, ABS(B.count- coalesce(t1.c1,0))+0.00001 as c1n from t0 "
            else:
                if is_avg_agg:
                    query += " (coalesce(t0.agg,0))+0.00001 as c0n,  (coalesce(t1.agg,0))+0.00001 as c1n from t0 "
                else:
                    query += "  ABS(coalesce(t0.c0,0))+0.00001 as c0n,  ABS(coalesce(t1.c1,0))+0.00001 as c1n from t0 "

            for t in ['t1']:
                query += 'join ' + t + ' on ' 
                for i in range(len(var)):
                    query += " t0.bucket_v{} = {}.bucket_v{} ".format(str(i),t ,str(i))
                    if i < len(var) - 1:
                        query += ' AND '
            query += ',A , B'

        else:
            if is_interv:
                if is_avg_agg:
                    query += " (A.agg * A.count - t0.agg * t0.c0 ) / ((A.count - coalesce(t0.c0,0))+0.00001) as c0n from A, t0"
                else:
                    query += " A.count - coalesce(t0.c0,0)+0.00001 as c0n from A, t0"
            else:
                if is_avg_agg:
                    query += " coalesce(t0.agg,0)+0.00001 as c0n from A, t0"
                else:
                    query += " coalesce(t0.c0,0)+0.00001 as c0n from A, t0"
            query += " where t0.pc0 <= {}".format(str(num_pred))

        query += ") "


        query += "select  "
        for i in range(len(var)):
            query += "case when bv{} = '-9999' then 'all' else CONCAT('(' , ROUND(CAST((v{}_max - v{}_min) / {} * (bv{} -1.0 ) + v{}_min as numeric), 3), ' , ', ROUND(cast((v{}_max - v{}_min) / {} * (bv{} ) + v{}_min as numeric), 3) ,')') end as rv{},".format(str(i), str(i), str(i), str(num_of_bucket),str(i),str(i), str(i), str(i), str(num_of_bucket), str(i), str(i), str(i))
        

        query += "cast(c0n AS float) / cast( c1n+0.00001 as float) as EValue from intervention, A, B, "
        for i in range(len(var)):
            query += " v{}_stat ".format(str(i))
            if i < len(var) - 1:
                query += ', '

        if not is_avg_agg:
            query += 'where ( c0n/A.count >= {} and  c0n/A.count <= {}) or ( c1n/B.count >= {} and  c1n/B.count <= {}) or ( c0n/A.count <= {} and  c1n/B.count >= {}) or ( c1n/B.count <= {} and c0n/A.count >= {})'.format(str(sel_min), str(sel_max), str(sel_min), str(sel_max), str(sel_min), str(sel_max),str(sel_min), str(sel_max))



        query += "order by evalue "
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

    else:
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

            query += " {} as agg, count(*) as c{}, ({}) as pc{} from {}".format(
                aggregate,
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
            query += "A as ( select {} as agg, count(*) from sample where {}),".format( aggregate,
                " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p0))))
            if num_queries == 2:
                query += "B as ( select {} as agg, select count(*) from sample where {}),".format(aggregate,
                    " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p1))))
        else:
            query += "A as ( select {} as agg, count(*) from {} where {}),".format(aggregate,
                table,
                " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p0)))
            )
            if num_queries == 2:
                query += "B as ( select {} as agg, count(*) from {} where {}),".format( aggregate,
                    table,
                    " and ".join(map(lambda x: "{} = '{}'".format(x[0], x[1]), zip(gb, p1)))
                )

        query += "intervention as (select "
        for a in range(num_queries):
            for i in range(len(var)):
                query += "t" + str(a) + ".v" + str(i) + " as " + "t" + str(a) + "v" + str(i) + ","

        if num_queries == 2:
            if is_interv:
                if is_avg_agg:
                    query += " (A.agg * A.count - t0.agg * t0.c0 ) / ((A.count - coalesce(t0.c0,0))+0.00001) as c0n, (B.agg * B.count - t1.agg * t1.c1 ) / ((B.count - coalesce(t1.c1,0))+0.00001)  as c1n from A,B, t0 left outer join t1 on  "
                else:
                    query += " A.count - coalesce(t0.c0,0)+0.00001 as c0n, B.count- coalesce(t1.c1,0)+0.00001 as c1n from A,B, t0 left outer join t1 on "
            else:
                if is_avg_agg:
                    query += " (coalesce(t0.agg,0))+0.00001 as c0n,  (coalesce(t1.agg,0))+0.00001 as c1n from A,B, t0 left outer join t1 on  "
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

    # def __init__(self, sql_query, db_raw, attrs, table, groupby_attr, uq1, uq2, uq_dir, topk=10, ppred=3, predicate_blacklist=None, range_exp=False):
    def __init__(self, sql_query, db_raw, attrs, table, groupby_attr, aggregate,uq1, uq2, uq_dir, topk=10, ppred=3, predicate_blacklist=None, range_exp=False, sel_min=1, sel_max=99):
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
        self.range_exp = range_exp
        self.sel_min = float(sel_min)
        self.sel_max = float(sel_max)
        self.aggregate = aggregate
        print("range_exp===========", self.range_exp)

    def find_explanation(self, interv=True):

        try:
            if self.uq2 == "None":
                if interv:
                    query = self.build_interv_query(self.attrs, self.aggregate ,self.uq1, None, None, self.dir,self.table, self.groupby_attr, 100, self.sel_min, self.sel_max)
                else:
                    query = self.build_aggrav_query(self.attrs, self.aggregate,self.uq1, None, None, self.dir, self.table,
                                                    self.groupby_attr, 100, self.sel_min, self.sel_max)
            else:
                if interv:
                    query = self.build_interv_query(self.attrs, self.aggregate,self.uq1, self.uq2, '/', self.dir,self.table, self.groupby_attr, 100, self.sel_min, self.sel_max)
                else:
                    query = self.build_aggrav_query(self.attrs, self.aggregate,self.uq1, self.uq2, '/', self.dir, self.table,
                                                    self.groupby_attr, 100, self.sel_min, self.sel_max)
            print(query)

            blacklist = []
            if self.predicate_blacklist is not None:
                for i in range(len(self.predicate_blacklist)):
                    blacklist.append([])
                    for j in range(len(self.predicate_blacklist[i])):
                        arr = self.predicate_blacklist[i][j].split(' = ')
                        for k in range(len(self.attrs)):
                            if self.attrs[k] == arr[0]:
                                if str(self.range_exp) == 'false':
                                    blacklist[-1].append("t{} = '{}'".format(str(k), arr[1]))
                                else:
                                    blacklist[-1].append("rv{} = '{}'".format(str(k), arr[1]))
                                break
            
            print("========blacklist========", blacklist)
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

    def build_interv_query(self, attrs, aggregate,q1, q2, op, dir, table, groupby_attr, sample_rate, sel_min, sel_max):
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
            equation = "cast( (c0n+0.0001) AS float) " + op + " cast( (c1n+0.0001) as float) as EValue";
            num_queries = 2

        interv_query = "with "
        interv_query += create_tables_helper(True, table, groupby_attr, aggregate,queries,
                                             var, equation, dir, num_queries, sample_rate, self.ppred, self.range_exp, sel_min, sel_max)
        # print(interv_query)
        return interv_query

    def build_aggrav_query(self, attrs, aggregate,q1, q2, op, dir, table, groupby_attr, sample_rate, sel_min, sel_max):

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

        aggrav_query += create_tables_helper(False, table, groupby_attr, aggregate,queries,
                                             var, equation, dir, num_queries, sample_rate, self.ppred, self.range_exp, sel_min, sel_max)

        return aggrav_query

