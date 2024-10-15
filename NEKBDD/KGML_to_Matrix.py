import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import itertools


def KGML_to_Matrix(pathway_name, KGML_file_path = "", save_path = ""):
    """
    Converts KGML file to adjacency matrix and node detail for KEGG human biological pathway analysis.

    Parameters:
    - pathway_name : the name of the pathway.
    - KGML_file_path : the file path of the KGML file.
    - save_path : the directory path where the output files will be saved.

    The function extracts node information and relationships from the KGML file,
    processes them, and outputs an adjacency matrix and node details as pandas DataFrames.
    """

    tree = ET.parse(KGML_file_path)
    root = tree.getroot()

    data_raw = []
    for child in root:
        data = {child.tag: child.attrib, "graphics": [], "component": [], "subtype": []}
        for children in child:
            data[children.tag].append(children.attrib)
        data_raw.append(data)

    for ii in range(len(data_raw)):
        data_raw[ii] = {k: v for k, v in data_raw[ii].items() if v}

    data_entry_raw, data_relation_raw = [], []
    for item in data_raw:
        if "entry" in item:
            data_entry_raw.append(item)
        if "relation" in item:
            data_relation_raw.append(item)

    for item in data_entry_raw:
        graphics = item.get("graphics", [{}])[0]
        if "name" in graphics:
            graphics["node_name"] = graphics.pop("name")
        if "type" in graphics:
            graphics["type1"] = graphics.pop("type")
    data_entry = [{**item["entry"], **item["graphics"][0]} for item in data_entry_raw]
    data_entry = pd.DataFrame(data_entry)[["id", "name", "type", "node_name"]]

    number = []
    for ii in range(len(data_entry_raw)):
        if "component" in data_entry_raw[ii]:
            number.append(len(data_entry_raw[ii]["component"]))
        else:
            number.append(int(1))
    component_id = []
    for ii in range(len(number)):
        for jj in range(number[ii]):
            component_id.append(range(len(number))[ii])

    data_entry_component = pd.DataFrame(data_entry.iloc[component_id ,:])

    component = []
    for ii in range(len(data_entry_raw)):
        if "component" not in data_entry_raw[ii]:
            component.append(float("NaN"))
        if "component" in data_entry_raw[ii]:
            for jj in data_entry_raw[ii]["component"]:
                component.append(jj["id"])

    component = pd.DataFrame(component, index = component_id)
    component.columns = ["component"]
    data_entry_new = pd.concat([data_entry_component, component], axis = 1)

    data_entry = data_entry[data_entry["type"] != "map"].reset_index(drop=True)

    unique_correspond = None
    unique_entryname = None
    unique_entryid = None

    unique_entryname = data_entry["name"].unique()
    unique_erntryid = data_entry["id"][data_entry["name"].duplicated() == False]

    match = lambda a, b: [ b.index(x) if x in b else None for x in a ]
    position_repeat = match(list(data_entry["name"][data_entry["name"].duplicated()]), list(data_entry["name"]))

    Var1 = list(data_entry["id"][data_entry["name"].duplicated()])
    Var2 = list(data_entry["id"][position_repeat])
    unique_correspond = {"Var1":Var1, "Var2":Var2}
    unique_correspond = pd.DataFrame(unique_correspond)

    data_relation = pd.DataFrame([item["relation"] for item in data_relation_raw])[["entry1", "entry2"]]

    if len(data_relation) == 0:
        return print("There are no relation in the ", pathway_name, "!")
    else:
        separate_from = list(data_entry_new[data_entry_new["type"] == "group"]["id"])
        separate_to = list(data_entry_new["component"][data_entry_new["component"].isnull() == False])

        relation_new = []
        for ii in range(len(data_relation.index)):
            if [i in data_relation.iloc[ii,].iloc[0] for i in separate_from].count(True) > 0:
                relation1 = [separate_to[i] for i, v in enumerate([i in data_relation.iloc[ii,].iloc[0] for i in separate_from]) if v == True]
            else:
                relation1 = [data_relation.iloc[ii,].iloc[0]]
            if [i in data_relation.iloc[ii, 1] for i in separate_from].count(True) > 0:
                relation2 = [separate_to[i] for i, v in enumerate([i in data_relation.iloc[ii, 1] for i in separate_from]) if v == True]

            else:
                relation2 = [data_relation.iloc[ii,].iloc[1]]
            relation_new.append([(x, y) for x in relation1 for y in relation2])
        relation_new = pd.concat(list(map(pd.DataFrame, relation_new)))
        relation_new.columns = ["Var1", "Var2"]
        relation_new = relation_new.reset_index(drop = True)

        def relation_group_fn(xx):
            sub_group = [separate_to[i] for i, v in enumerate([i in xx for i in separate_from]) if v == True]
            return pd.DataFrame(list(itertools.combinations(sub_group, 2)))

        if len(separate_from) > 0:
            relation_group = pd.concat(list(map(pd.DataFrame, list(map(relation_group_fn, sorted(list(set(separate_from))))))))
            relation_group.columns = ["Var1", "Var2"]
            relation_group = relation_group.reset_index(drop = True)
            Reverse_relation = relation_group[["Var2", "Var1"]]
            Reverse_relation.columns = ["Var1", "Var2"]
            relation_group = pd.concat([relation_group, Reverse_relation]).reset_index(drop = True)
            relation_new = pd.concat([relation_new, relation_group]).reset_index(drop = True)

        relationship = pd.DataFrame(np.zeros((len(data_entry["id"]), len(data_entry["id"])), dtype = int), index = list(data_entry["id"]), columns = list(data_entry["id"]))
        position1 = match(relation_new["Var1"], list(data_entry["id"]))
        position2 = match(relation_new["Var2"], list(data_entry["id"]))
        relation_position = pd.DataFrame({"position1" :position1, "position2" :position2})
        for ii in range(len(relation_position.index)):
            x = relation_position.iloc[ii, 0]
            y = relation_position.iloc[ii, 1]
            relationship.iloc[x, y] = 1

        pos1 = match(unique_correspond["Var1"], list(data_entry["id"]))
        pos2 = match(unique_correspond["Var2"], list(data_entry["id"]))

        if len(pos1) != 0:
            for ii in range(len(pos1)):
                relationship.iloc[pos2[ii], ] = relationship.iloc[pos2[ii], ] + relationship.iloc[pos1[ii], ]
                relationship.iloc[:, pos2[ii]] = relationship.iloc[:, pos2[ii]] + relationship.iloc[:, pos1[ii]]
        if len(pos1) > 0:
            pos1_relationship = [relationship.columns[ii] for ii in pos1]
            relationship = relationship.drop(pos1_relationship, axis = 1)
            relationship = relationship.drop(pos1_relationship, axis = 0)

            data_entry.index = list(data_entry["id"])
            entry_pos1 = [data_entry["id"].iloc[ii] for ii in pos1]
            node_detail = data_entry.drop(entry_pos1, axis = 0)
            node_detail = node_detail[["name", "type", "node_name"]]

            if list(data_entry["type"] == "group").count(True) != 0:
                delete_group = data_entry.drop(entry_pos1, axis = 0)
                delete_group_drop = [delete_group.index[ii] for ii, vv in enumerate(delete_group["type"] == "group") if vv == True]
                relationship = relationship.drop(delete_group_drop, axis = 0)
                relationship = relationship.drop(delete_group_drop, axis = 1)
                node_detail = node_detail.drop(delete_group_drop, axis = 0)
        else:
            data_entry.index = list(data_entry["id"])
            node_detail = data_entry[["name", "type", "node_name"]]

        for ii in range(len(relationship.index)):
            for jj in range(len(relationship.columns)):
                if relationship.iloc[ii, jj] > 1:
                    relationship.iloc[ii, jj] = 1
        adj_matrix = relationship

        file_name = save_path + pathway_name + "(directed)"
        adj_matrix.to_pickle(file_name)
        file_name = save_path + pathway_name + "(node_detail)"
        node_detail.to_pickle(file_name)
        return "Success"
