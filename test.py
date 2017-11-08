import decition_tree_utils as gt
import DecisionTree as dt
import random

#
# table_set = gt.import_training_data_file("data_set", True)
# print(gt.get_label(table_set[0], 9))
# id_set = table_set[1]
# table = table_set[0]
# entrophy = gt.get_entrophy(table, id_set)
# weak_set = set([0,2,3,4,7,8,9,12])
# info_weak = gt.get_entrophy(table,weak_set)
# strong_set = set([1,5,6,10,11,13])
# info_strong = gt.get_entrophy(table,strong_set)
#
# print(entrophy - 6*info_strong/14 - 8*info_weak/14)
#
# ###############################################################
# # attribute split test
# # split_mapping = gt.split_by_att(table, id_set, 0)
# # print(split_mapping)
# ###############################################################
# # test information gain function
# ig = gt.get_ig(table, id_set, 2, entrophy)
# print(ig)
# ###############################################################
# test get att set
#att_set = table_set[2]
#print(att_set)
# ###############################################################
# test get majority label
# maj_label = dt.get_majority_label(table, id_set)
# print(maj_label[0], maj_label[1])
# ##############################################################
# test model building
# decision_tree_root = dt.build_model(table, id_set, att_set)
# decision_tree_root = dt.build("data_set", True)
# dt.bfs(decision_tree_root)
# ##############################################################
# dictionary test
# dict = {'a': 'hello', 'b': 'world'}
# print(dict)
# x = dict.popitem()
# dict[x[0]] = x[1]
# print(x[1])
# print(dict)
# ###############################################################
# test single record prediction
# decision_tree_root = dt.build("data_set", True)
# pred = dt.predict(decision_tree_root, ['Rain','Mild','High','Weak'])
# print(pred)
# ###########################################################
# test dictionary contains
# dict ={1: 'asa', 3: 'sasdsdsa'}
# ##############################################################
# test random
# for i in range(10):
#     print(random.randrange(1, 10))
# #############################################################
# test precision
decision_tree_root = dt.build("test_set.csv", True)
print(dt.predict(decision_tree_root, [1,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,0,0,0,1,1]))
dt.precision(decision_tree_root,"test_set2.csv", True)
# #############################################################
# pruning test
print("######################## After Pruning #################")
dt.prune(decision_tree_root)
dt.precision(decision_tree_root,"test_set2.csv", True)
print("######################## After Pruning #################")
dt.prune(decision_tree_root)
dt.precision(decision_tree_root,"test_set2.csv", True)