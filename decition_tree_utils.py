import numpy as np


def import_training_data_file(file_name, remove_first_line): # helper function to read training data file
    file = open(file_name)
    table = {}
    id = 0
    id_set = set()
    lebel_set = set()

    start_line = 0
    if remove_first_line is True:
        start_line = 1

    for line in file.readlines()[start_line:]:
        # print(line)
        line = line.replace('\n', '')
        table[id] = line.split(',')
        id_set.add(id)
        id += 1

    file.close()

    for i in range(len(table[0]) - 1):
        lebel_set.add(i)
    # @return input data table
    # @return number of attributes
    return table, id_set, lebel_set


# calculate entrophy by giving table and id_set
def get_entrophy(table, id_set):

    label_map = {}

    for id in id_set:
        line = table.get(id)
        label = line[len(line) - 1]
        cur_count = label_map.get(label)
        if cur_count is None:
            cur_count = 0
        cur_count += 1
        label_map[label] = cur_count

    # print(label_map)
    # print(len(id_set))
    entrophy = calculate_entrophy(label_map,len(id_set))
    return entrophy


# helper function to calculate entrophy
def calculate_entrophy(label_map, num_cat):
    entrophy = 0

    # print(label_map)
    # print(num_cat)

    for lb in label_map:
        cur_portion = np.float32(label_map[lb]) / np.float32(num_cat)
        entrophy += -cur_portion * np.log2(cur_portion)

    return entrophy


# choose attribute with largest information gain
# att_set is the att set left to split on
def choose_attribute(table, id_set, att_set, original_entrophy):
    max_information_gain = 0
    att_result = None

    for att in att_set:
        cur_IG = get_ig(table, id_set, att, original_entrophy)
        if cur_IG >= max_information_gain:
            max_information_gain = cur_IG
            att_result = att

    return att_result


# calculate information gain
def get_ig(table, id_set, att, original_entrophy):
    attcat_idset_mapping = split_by_att(table, id_set, att)
    sum_each_entrophy_mult_portion = 0

    if len(attcat_idset_mapping) == 0:
        return 0

    for attcat in attcat_idset_mapping:
        idset = attcat_idset_mapping[attcat]
        portion = np.float32(len(idset)) / np.float32(len(id_set))
        cur_entrophy = get_entrophy(table, idset)
        sum_each_entrophy_mult_portion += portion * cur_entrophy

    return original_entrophy - sum_each_entrophy_mult_portion


# helper function to splits id_sets basing on current id_set and attribute
def split_by_att(table, cur_id_set, att):
    # maintain a dictionary with <att_x, (id set)>
    attribute_category_idset_mapping = {}

    for cur_id in cur_id_set:
        cur_att_cat = table.get(cur_id)[att]
        if attribute_category_idset_mapping.get(cur_att_cat) is None:
            attribute_category_idset_mapping[cur_att_cat] = []
        attribute_category_idset_mapping[cur_att_cat].append(cur_id)

    return attribute_category_idset_mapping


# helper function to get corresponding label given the data record id
def get_label(table, cur_id):
    return table.get(cur_id)[-1]