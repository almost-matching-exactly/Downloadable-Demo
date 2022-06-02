from .EnsembleLearner import EnsembleLearner
import logging

class AttributeRecommender:

    # constant learning rate
    adjust_rate = 0.05

    def __init__(self, db_raw):
        # self.sql_query = sql_query
        self.error_message = None
        self.cur = db_raw
        self.weights = [0.7, 0.3]
        # recommendation scores for AMS, RF
        self.last_score1 = []
        self.last_score2 = []
        # whether there is only one value in the user question
        self.last_one_var = False

    def recommend_attributes(self,
                             sql_query, attr_list, sql_select, sql_from, groupby,
                                      q1, q2, q_dir, b_list, rec_k_attr):
        input_string, used_attr = EnsembleLearner.build_query(attr_list, q1, q2, sql_select, sql_from, groupby, b_list)
        result = self.cur.execute(input_string).fetchall()
        print(input_string, result)
        input_data = []
        input_label = []
        all_attr = groupby + used_attr
        for r in result:
            input_data.append(list(map(lambda x: all_attr[x[0]] + ' : ' + str(x[1]).strip(), enumerate(list(r)))))
            input_label.append(' --- '.join(list(map(lambda x: str(x).strip(), list(r)[:len(groupby)]))))
        # print("input_string", input_string)
        # print("result", result)
        q = [q1, q2]
        attr_score1 = dict()
        attr_score2 = dict()

        self.last_one_var = q2 == 'None' or q2 is None

        print(q1, q2)

        if q2 == 'None' or q2 is None:
            learner = EnsembleLearner(True, groupby, q, input_data)
            res1, _ = learner.compute_classifier_results()
        else:
            learner = EnsembleLearner(False, groupby, q, input_data, input_label)
            res1, res2 = learner.compute_classifier_results()
            for i, r in enumerate(res2):
                attr_score2[used_attr[i]] = r
        
        # print(res1, res2)

        total_score = 0
        for r in res1:
            score = r[3]
            lhs = r[0].lhs
            rhs = r[0].rhs
            for a_str in lhs:
                arr = a_str.split(' : ')
                if arr[0] in groupby:
                    continue
                if arr[0] not in attr_score1:
                    attr_score1[arr[0]] = 0
                attr_score1[arr[0]] += score
                total_score += score
        for a in attr_score1:
            attr_score1[a] /= total_score

        logging.debug(attr_score1)
        logging.debug(attr_score2)
        res = []
        if q2 == 'None' or q2 is None:
            for i, a in enumerate(used_attr):
                res.append([a, attr_score1[a]])
        else:
            for i, a in enumerate(used_attr):
                if a not in attr_score1:
                    attr_score1[a] = 0
                if a not in attr_score2:
                    attr_score2[a] = 0
                res.append([a, self.weights[0] * attr_score1[a] + self.weights[1] * attr_score2[a]])
                # res.append([a, self.weights[0] * attr_score1[a]/total_score + self.weights[1] * attr_score2[a]])


        res = sorted(res, key=lambda x: x[1], reverse=True)
        
        
        self.last_score1 = attr_score1
        self.last_score2 = attr_score2
        print("========last_score1========")
        print(self.last_score1, self.last_score2)
        return res
