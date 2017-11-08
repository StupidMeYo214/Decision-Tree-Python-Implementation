import decition_tree_utils as gt
import queue
import numpy as np
import copy
import random


# encapsulate model building function
def build(file_name, remove_first_line):
    table_set = gt.import_training_data_file(file_name, remove_first_line)
    table = table_set[0]
    id_set = table_set[1]
    # print(len(id_set))
    att_set = table_set[2]
    # print(att_set)
    return build_model(table, id_set, att_set)


# prune randomly in a decision tree
def prune(model):
    node_list = []
    collect_all_nodes(model, node_list)
    num_nodes = len(node_list)
    if num_nodes == 0:
        print("Cannot prune anymore")
    else:
        rand_index = random.randrange(1, num_nodes)
        print('index == ', rand_index)
        node_list[rand_index].is_leaf = True


# using dfs to collect all decision tree nodes
def collect_all_nodes(model, node_list):
    if model.is_leaf is True:
        return model

    node_list.append(model)
    children = model.children

    for child_att in children:
        child = children[child_att]
        collect_all_nodes(child, node_list)
    return model


# predict single record
def predict(model, data):
    # base case
    if model.is_leaf is True:
        return model.majority_label

    children = model.children
    split_att = model.split_att
    next_node = None
    split_att_in_data = data[split_att]

    if split_att_in_data in children:
        next_node = children[split_att_in_data]
    else:
        pop = children.popitem()
        children[pop[0]] = pop[1]
        next_node = pop[1]

    return predict(next_node, data)


# calculate precision by giving validation set file name
def precision(model, validation_set_file_name, remove_first_line):
    table = gt.import_training_data_file(validation_set_file_name, remove_first_line)[0]
    size = len(table)
    correct = 0

    for cur_id in table:
        cur_data = table.get(cur_id)
        cur_predict = predict(model, cur_data)
        if cur_predict == gt.get_label(table, cur_id):
            correct += 1

    print("validation set size is ", size, ", the model predicted ", correct, " records data correctly")
    accuracy = np.float32(correct) / np.float32(size)
    print("precision is ", accuracy)
    return  accuracy


# build model using recursion
def build_model(table, id_set, att_set):
    maj_label_and_ispure = get_majority_label(table, id_set)
    if len(att_set) == 0 or maj_label_and_ispure[1] is True:    # base case 1.no att 2.label is pure
        # print(att_set, maj_label_and_ispure)
        return Node(True, None, maj_label_and_ispure[0], None)

    cur_entrophy = gt.get_entrophy(table, id_set)
    split_att = gt.choose_attribute(table, id_set, att_set, cur_entrophy)
    # print(level,split_att)

    attribute_category_idset_mapping = gt.split_by_att(table, id_set, split_att)
    # print(attribute_category_idset_mapping)

    children_dict = {}
    att_set.discard(split_att)

    for cur_att in attribute_category_idset_mapping:
        new_att_set = copy.deepcopy(att_set)
        cur_id_set = attribute_category_idset_mapping.get(cur_att)
        # child = create_model(table, cur_id_set, att_set, level + 1)
        child = build_model(table, cur_id_set, new_att_set)
        children_dict[cur_att] = child

    return Node(False, split_att, maj_label_and_ispure[0], children_dict)


# data structure for decition tree
class Node:
    # node type definition
    is_leaf = False
    # split attribute
    split_att = None
    # maintain a dictionary to map <attribute, children_node>
    children = {}
    # maintain majority to prepare for pruning & classification
    majority_label = None

    def __init__(self, is_leaf, split_att, majority_label, children):
        self.is_leaf = is_leaf
        self.split_att = split_att
        self.majority_label = majority_label
        self.children = children


# get majority label and id_set
def get_majority_label(table, id_set):
    label_map = {}
    maj_label = None
    max_count = 0
    for cur_id in id_set:
        cur_label = gt.get_label(table, cur_id)
        cur_count = label_map.get(cur_label)
        if cur_count is None:
            cur_count = 0
        cur_count += 1
        label_map[cur_label] = cur_count

    for key in label_map:
        # print(key, label_map[key])
        if label_map[key] >= max_count:
            maj_label = key

    return maj_label, len(label_map) == 1