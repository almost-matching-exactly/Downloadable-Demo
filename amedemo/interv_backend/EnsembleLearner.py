from efficient_apriori import apriori
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


class EnsembleLearner:

    def __init__(self, one_variable, gb, q, input, labels=None):


        if one_variable:
            self.numClassifiers = 1
        else:
            self.numClassifiers = 2

        self.gb = gb
        self.q = q
        self.input = input
        self.labels = labels
        print("EnsembleLearner q:", q)
        self.q1 = q[0].strip().split(" --- ")[:-1]
        self.q2 = q[1].strip().split(" --- ")[:-1]
        # print("q2", self.q2)
        # ASM score = confidence * w + support * (1-w)
        self.w = 0.5

    @staticmethod
    def build_query(attrs, q1, q2, sql_select, sql_from, gb, blacklisted_variables):
        # print("build_query:",attrs, q1, q2, sql_select, sql_from, gb, blacklisted_variables)

        query = ["select "]
        p0 = ["", ""] # hard coded for 2 variables
        p1 = ["", ""]
        # whitelisted_attrs = []
        # blacklisted attributes
        BLLIST = ['rated_at', 'release_date']

        # for i in range(len(attributes)):
        #     if attributes[i] not in blacklisted_variables:
        #         whitelisted_attrs.append(attributes[i])
        whitelisted_attrs = attrs

        q2Exists = q2 is not None and q2 != 'None'
        # predicate of each user question tuple
        p0 = q1.strip().split(" --- ")
        print('p0', p0)

        if q2Exists:
            p1 = q2.strip().split(" --- ")

        ''' example query:
            select
        shipmode, region, rowid, orderid, orderdate, orderpriority, orderquantity, sales, discount,
        profit, unitprice, shippingcost, customername, province, region, customersegment, productcategory,
        productsubcategory, productname, productcontainer, productbasemargin, shipdate
        from superstore where (shipmode = Delivery Truck and shipmode = Atlantic)
        '''

        if q2Exists:
            for i in range(len(gb)):
                query.append(gb[i] + ", ")

        else:
            query.append(gb[0] + ", ")

        sj = []
        used_attr = []
        for i in range(len(whitelisted_attrs)):
            if len(gb) > 1:
                if whitelisted_attrs[i] in gb:
                    '''Skip.Note,
                    if the query variable exists in this partition, then the SQL query will have fewer attributes than the width.
                    '''
                else:
                    if whitelisted_attrs[i] not in BLLIST and not whitelisted_attrs[i].startswith('genres_'):
                        sj.append(whitelisted_attrs[i])
                        used_attr.append(whitelisted_attrs[i])
            else:
                if whitelisted_attrs[i] in gb:
                    '''Skip.Note,
                    if the query variable exists in this partition, then the SQL query will have fewer attributes than the width.
                    '''
                else:
                    if whitelisted_attrs[i] not in BLLIST and not whitelisted_attrs[i].startswith('genres_'):
                        sj.append(whitelisted_attrs[i])
                        used_attr.append(whitelisted_attrs[i])

        # for sampling
        # did not finish
        # tenThousandRowsAsPercent = 10000.0 / nrows * 100
        tenThousandRowsAsPercent = 100.0
        query.append(', '.join(sj))
        query.append(" from " + sql_from)
        if tenThousandRowsAsPercent < 100.0:
            query.append(" tablesample system (" + f"{tenThousandRowsAsPercent:.9f}" + ") where ")
        else:
            query.append(" where ")

        query.append("(")

        sj = []
        if len(gb) >= 2:
            for i in range(len(gb)):
                sj.append(gb[i] + " = '" + p0[i] + "'")

        else:
            # sj.append(gb[0] + " = '" + q1 + "'")
            sj.append(gb[0] + " = '" + p0[0] + "'")

        query.append(' and '.join(sj))
        query.append(")")

        if q2Exists:
            query.append(" or ")
            query.append("(")
            sj = []
            if len(gb) == 1:
                # sj.append(gb[0] + " = '" + q2 + "'")
                sj.append(gb[0] + " = '" + p1[0] + "'")
            else:
                for i in range(len(gb)):
                    sj.append(gb[i] + " = '" + p1[i] + "'")
            query.append(' and '.join(sj))
            query.append(") ")

        return ' '.join(query), used_attr

    def compute_classifier_results(self):

        if self.numClassifiers == 1:
            return self.compute_association_rule_variables(), None
        else:
            as_res = self.compute_association_rule_variables()
            rf_res = self.compute_random_forest_variables()
            return as_res, rf_res

    def compute_association_rule_variables(self):
        itemsets, rules = apriori(self.input, min_support=0.2, min_confidence=0.5, max_length=4)
        res = []
        for r in rules:
            flag = 0
            rhs = list(map(lambda x: str(x).strip(), list(r.rhs)))

            for k in range(len(self.q1)):
                if self.q1[k] in rhs:
                    flag += 1
            # at most one of the righthand side attributes can be out of the user question tuple
            if flag >= len(rhs)-1:
                res.append([r, r.confidence, r.support, self.w * r.confidence + (1-self.w) *r.support])
        return res

    def compute_random_forest_variables(self):
        df = pd.DataFrame(self.input)
        X = df.iloc[:, len(self.gb):].values
        y = pd.DataFrame(self.labels).values

        print(self.input, self.gb)
        print(X,y)

        # transform groups to labels
        les = []
        for i in range(X.shape[1]):
            le = preprocessing.LabelEncoder()
            le.fit(X[:, i])
            les.append(le)
            X[:, i] = le.transform(X[:, i])
        le = preprocessing.LabelEncoder()
        # print(df)
        le.fit(y)
        les.append(le)
        y = le.transform(y)

        forest = RandomForestClassifier(n_estimators=10, max_depth=10)
        print("\nX:", X[:10])
        print("\ny:",y[:10])
        forest.fit(X, y)
        importances = forest.feature_importances_
        print(importances)

        return importances



