import random

import pandas as pd
import os
import statistics
import json
import urllib.request as urlreq
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bio as dashbio
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import math
import sys
import numpy as np
import plotly.express as px

app = Dash(__name__)

# define configs to customize graph
layout_config = {

    "labels": {"display": False},
    "ticks": {"display": False}
}

text_config = {
    "innerRadius": 1.02,
    "style": {
        "font-size": 12,
    },
}

chords_config = {"color": "Spectral",
                 # Spectral is essentially the color scale (from https://github.com/d3/d3-scale-chromatic) that we are configuring the chords to use. (This is what the 'value' parameter represents)
                 "opacity": 0.8,
                 'tooltipContent': {
                     'source': 'source',
                     'sourceID': 'id',
                     'target': 'target',
                     'targetID': 'id',
                     'targetEnd': 'end'},
                 }


def min_max_scaler(data, feature_range):
    """

    :param data: data list to be scaled
    :param feature_range: range to be scaled to (e.g (0, 1))
    :return: scaled list of data
    """
    # standard min-max scaling
    if len(data) == 0:
        return []
    scaler = MinMaxScaler(feature_range=feature_range)
    # transform data
    scaled = scaler.fit_transform(data)
    return scaled


def get_text(input_dict):
    """

    :param input_dict: input dictionary which is referenced to get node information
    :return: dictionary containing text information for graph
    """
    # configures the text
    # mainly makes sure that the text goes OUTSIDE of the nodes

    text_info = []
    for node in input_dict['Nodes']:
        text_info.append(
            {
                "block_id": node['id'],
                "position": node['len'] / 2,
                "value": node['id']  # [:len(node['id'])-2]

            }
        )
    """
    Should be in format:
    data = [
  {
    "block_id": "chr1",
    "position": 1150000,
    "value": "p36.33",
  },
  ...
]
    """

    return text_info


def get_length_of_node(node_id, input_dict):
    """

    :param node_id: id of node that length is requested for
    :param input_dict: dictionary containing all nodes and chords; will be referenced to return result
    :return: length of given node
    """
    # references our input dictionary to return the length of the specified node
    for node in input_dict['Nodes']:
        if node['id'] == node_id:
            return node['len']
    return 0


def reverse(lst):
    """

    :param lst: input list to be reversed
    :return: reversed list
    """
    new_lst = lst[::-1]
    return new_lst


def create_node_length(original_df, call_stack):
    """
    Node length is calculated by the (runtime of the callstack)/(cumulative runtime of all the call stacks shown in that given run)

    :param original_df: dataframe containing information to calculate node length
    :param call_stack: call stack that represents node to calculate proper length
    :return: appropriate node length
    """
    # calculate total_runtime
    combined_func_df = pd.read_csv("filtered_inp_files/func_data/combined_func_data.csv")
    total_runtime = 0
    for chosen_call_stack in combined_func_df["call_stack"]:
        for i in range(len(original_df['call_stack'])):
            if chosen_call_stack == original_df['call_stack'][i]:
                total_runtime += original_df['exclusive_runtimes'][i]

    # now get individual ratio based on call_stack parameter
    call_stack_runtime = 0
    for i in range(len(original_df['call_stack'])):
        if call_stack == original_df['call_stack'][i]:
            call_stack_runtime += original_df['exclusive_runtimes'][i]
    return call_stack_runtime / total_runtime


def remove_empty_nodes(original_dict):
    """
    Removes empty nodes (nodes without chords) from given original_dict

    :param original_dict: dictionary which contains redundant nodes
    :return: dictionary with empty nodes removed
    """
    nodes_to_delete = []
    for node in original_dict["Nodes"]:
        if check_num_connections(node["id"], original_dict)[0] == 0:
            nodes_to_delete.append(node["id"])

    index = 0
    for i in range(len(original_dict["Nodes"])):
        if original_dict["Nodes"][index]['id'] in nodes_to_delete:
            del original_dict["Nodes"][index]
            index -= 1
        index += 1


def get_nodes(input_dict, file_name_limit):
    """
    Calculates nodes for vis

    :param input_dict: dict where nodes will be added to
    :param file_name_limit: delimiter to access files (if run name is 39, file_delimiter will be 2; if run name is 104, file_delimeter will be 3; etc.)
    :return: input dict with all node information added
    """

    print("Getting Node Info...\n")
    """
    id:
    label:
    color:
    len:
    """
    # get dictionary that represents number of calls of certain call stack

    # will retain order with corresponding call_stack

    # color_list = ["#232023", "#A7A6BA", "#808080", "#C5C6D0", "#D3D3D3"]
    color_list = ["#00B571", "#5D7B70", "#F633FF", "#335BFF", "#33FFB2"]
    index = 0
    for i in os.listdir("filtered_inp_files/only_rem_duplicates/"):
        node_length_list = []
        original_df = pd.read_csv(
            "filtered_inp_files/only_rem_duplicates/" + str(
                os.listdir("filtered_inp_files/only_rem_duplicates/")[index]))
        combined_func_df = pd.read_csv("filtered_inp_files/func_data/combined_func_data.csv")

        if index % 2 == 0:
            for call_stack in combined_func_df['call_stack']:
                if create_node_length(original_df, call_stack) != 0:
                    input_dict['Nodes'].append({
                        "id": str(call_stack) + "_" + i[0:file_name_limit],
                        "label": str(call_stack) + "_" + i[0:file_name_limit],
                        "color": color_list[index],
                        # "len" : n_calls_dict[call_stack]
                        "len": create_node_length(original_df, call_stack)

                    })
                    node_length_list.append(create_node_length(original_df, call_stack))

        else:

            for call_stack in reverse(combined_func_df['call_stack'].tolist()):
                if create_node_length(original_df, call_stack) != 0:
                    input_dict['Nodes'].append({
                        "id": str(call_stack) + "_" + i[0:file_name_limit],
                        "label": str(call_stack) + "_" + i[0:file_name_limit],
                        "color": color_list[index],
                        "len": create_node_length(original_df, call_stack)
                    })
                    node_length_list.append(create_node_length(original_df, call_stack))

        index += 1
    return input_dict


def average(lst):
    """

    :param lst: list for average to be calculated
    :return: average of list
    """
    return sum(lst) / len(lst)


def get_chords(input_dict, file_name_dict):
    """
    calculate chords for vis
    :param input_dict:
    :param file_name_dict:
    :return: input dictionary with all chord info added
    """

    print("Getting Chord Info...\n")

    for json_dump_index in range(len(os.listdir("filtered_inp_files/only_rem_duplicates/"))):
        # go through chord data folder that is categorized by JSON Dump
        print("Going through " + str(os.listdir("filtered_inp_files/only_rem_duplicates/")[json_dump_index]))

        # get the JSON dumps to compare the previous one with
        json_dumps_to_compare_with = os.listdir("filtered_inp_files/only_rem_duplicates/")
        del json_dumps_to_compare_with[json_dump_index]

        original_df = pd.read_csv(
            "filtered_inp_files/only_rem_duplicates/" + str(
                os.listdir("filtered_inp_files/only_rem_duplicates/")[json_dump_index]))

        for i in range(len(original_df)):

            source_start = 0
            total_possible_connections = len(json_dumps_to_compare_with)
            num_chords_from_specific_node = 0

            for j in range(len(json_dumps_to_compare_with)):

                # loop is used to compare our top functions (specified by user input) with the other JSON Dumps to find matches

                # target start indicates where on the target node the chord will "start"

                # create and configure a dataframe that contains the JSON dump that we are comparing our df_by_severity to
                comparing_to_df = pd.read_csv(
                    "filtered_inp_files/only_rem_duplicates/" + str(json_dumps_to_compare_with[j]))

                # scale runtimes
                original_df["scaled_runtimes"] = min_max_scaler(
                    np.reshape(original_df["exclusive_runtimes"].tolist(), (-1, 1)), (0.1, 1))
                comparing_to_df["scaled_runtimes"] = min_max_scaler(
                    np.reshape(comparing_to_df["exclusive_runtimes"].tolist(), (-1, 1)), (0.1, 1))

                # go through df_by_severity to compare and find similarities in other datasets

                for k in range(len(comparing_to_df)):

                    # go through comparing_to_df to compare with df_by_severity

                    if (original_df['call_stack'].tolist()[i] ==
                            comparing_to_df['call_stack'].tolist()[k]):
                        num_chords_from_specific_node += 1

                        # check if there is a connection: one function in our source dataframe/JSON dump is found in another dump
                        # this indicates that a chord needs to be drawn

                        # create the chord based on this connection

                        source_id = original_df['call_stack'].tolist()[i] + "_" + file_name_dict[
                            "filtered_inp_files/only_rem_duplicates/" + str(
                                os.listdir("filtered_inp_files/only_rem_duplicates/")[json_dump_index])]
                        target_id = comparing_to_df['call_stack'].tolist()[k] + "_" + file_name_dict[
                            "filtered_inp_files/only_rem_duplicates/" + json_dumps_to_compare_with[j]]
                        # assign corresponding target and source id

                        """
                        we know the increment has to be this, regardless of the runtime because what we are doing is : (nodewidth*runtime_1)/(runtime_1+runtime_2) where runtime_1=runtime_2=runtime_3
                        This comes out to node_width/2, and 2 is the number of total possible connections
                        """
                        source_increment = 0.75 * get_length_of_node(source_id, input_dict) / (
                            total_possible_connections)

                        target_increment = 0.75 * get_length_of_node(target_id, input_dict) / (
                            total_possible_connections)

                        # set up a proper source increment, so that all the chords will fit on the nodes, and there will be a little bit of space between each chord for readability

                        target_start = get_length_of_node(target_id, input_dict) - target_increment

                        input_dict["Chords"].append({
                            "source": {

                                "id": source_id,
                                "start": source_start,
                                "end": source_start + source_increment
                            },
                            "target": {
                                "id": target_id,
                                "start": target_start,
                                "end": target_start + target_increment
                            },
                            "value": 0.5,
                            "call_stack": source_id,
                            "source_severity": original_df['sev_bin_upper_bound'][i],
                            "target_severity": comparing_to_df['sev_bin_upper_bound'][k]

                        })
                        source_start += source_increment
                        source_start += (get_length_of_node(source_id, input_dict) / (total_possible_connections)) - (
                                    0.75 * get_length_of_node(source_id, input_dict) / (total_possible_connections))

                        target_start -= target_increment
                        target_start -= (get_length_of_node(target_id, input_dict) / (total_possible_connections)) - (
                                    0.75 * get_length_of_node(target_id, input_dict) / (total_possible_connections))

    return input_dict


def color_chords(input_dict):
    """
    colors the chords depending on how many connections they represent
    :param input_dict: dict to be modified
    :return: dict with chord colors updated
    """
    # assign color values based on number of connections for graphs 1 and 2 (individual, independent graphs will be evaluated in the filter_nodes function)
    value_list = [1.0, 0.3, 0.1]
    all_node_id = []
    for dict in input_dict["Nodes"]:
        all_node_id.append(dict['id'])

    for id in all_node_id:
        counter_for_func = 0
        for dict in input_dict["Chords"]:
            if dict["source"]["id"] == id:
                counter_for_func += 1
        for dict in input_dict["Chords"]:
            if dict["source"]["id"] == id:
                dict["value"] = value_list[counter_for_func - 1]

    return input_dict


def check_num_connections(node_id, input_dict):
    """

    :param node_id: input node to check number of connections for
    :param input_dict: dict to reference
    :return: number of connections in input dict for given node id
    """
    num_connections = 0
    resulting_id = []
    for chord in input_dict["Chords"]:
        if chord["source"]["id"] == node_id and chord["source"]["end"] - chord["source"]["start"] != 0.000000005:
            num_connections += 1
            resulting_id.append(chord["target"]["id"])

    return num_connections


def filter_nodes(original_dict, num_connections):
    """
    this function filters the nodes based on the number of connections, which is used to create the independent graphs
    :param original_dict: dict to reference
    :param num_connections: number of connections to filter by
    :return:
    """
    value_list = [1.0, 0.3, 0.1]  # will need to be extended if more than 4 dumps are being compared
    # filters nodes by number of connections
    nodes_to_delete = []
    for node in original_dict["Nodes"]:
        if num_connections != check_num_connections(node["id"], original_dict):
            nodes_to_delete.append(node["id"])
    index = 0
    for i in range(len(original_dict["Nodes"])):
        if original_dict["Nodes"][index]['id'] in nodes_to_delete:
            del original_dict["Nodes"][index]
            index -= 1
        index += 1

    index = 0

    # try:
    if len(original_dict["Nodes"]) != 0:

        for i in range(len(original_dict["Chords"])):
            if original_dict["Chords"][index]["source"]["id"] in nodes_to_delete:
                del original_dict["Chords"][index]
                index -= 1
            else:
                original_dict["Chords"][index]["value"] = value_list[num_connections - 1]
            index += 1

    return original_dict


def filter_chords(input_dict):
    # this function goes through the dict and removes redundant chords
    """
    removes redundant chords from input dict

    :param input_dict: dict to modify
    :return: updated dict without redundant chords
    """
    value_list = [1.0, 0.3, 0.1]
    """
    1.0 = 1 connection
    0.3 = 2 connections
    0.1 = 3 connections
    """

    # remove empty chords: #empty chords are identified with having the value 0.5 and are still in the list

    for i in range(len(input_dict["Chords"])):
        counter = 0
        index = 0
        if input_dict["Chords"][i]["source"]["end"] == 0.000000005:
            break
        if input_dict["Chords"][i]["value"] in value_list:
            num_connections = value_list.index(input_dict["Chords"][i]["value"]) + 1
            # if there is 1 connection, then there will be 1 redundant chord that needs to be deleted
            current_source_id = input_dict["Chords"][i]["source"]["id"]
            current_target_id = input_dict["Chords"][i]["target"]["id"]
            for j in range(i, len(input_dict["Chords"])):

                if input_dict["Chords"][j]["source"]["id"] == current_target_id and input_dict["Chords"][j]["target"][
                    "id"] == current_source_id:
                    input_dict["Chords"][i]["source"]["end"] = float(input_dict["Chords"][i]["source"][
                                                                         "start"]) + 0.000000006  # this represents a formerly redundant chord
                    input_dict["Chords"][i]["target"]["end"] = float(input_dict["Chords"][i]["target"][
                                                                         "start"]) + 0.000000006  # this is a different number, so you can distinguish which chord was shortened to remove duplicates versus chords that were added to change color
                    # we need to keep the redundant chords in because the color calculation won't work if they are removed
                    counter += 1
                index += 1

    return input_dict


if __name__ == '__main__':

    """

        Command Line Params:

        python alternate_vis.py force_compute file_name_limit

        force_compute (optional): -f means that it will recalculate chords, nodes, etc. If not included, program will use previously cached information


    """
    argv = sys.argv
    del argv[0]

    if "-f" in argv:  # force compute
        print("RECOMPUTING DATA SINCE -f FLAG WAS FOUND...\n")

        input_dict = {"Nodes": [], "Chords": []}
        """
        Blue: "#797EF6"
        Orange: "#FF781F"
        Red: "#FF781F"
        """

        #file_name_limit = int(argv[1])



        file_name_dict = {}
        for i in os.listdir("filtered_inp_files/only_rem_duplicates/"):
            file_name_limit=0
            for j in i:
                if j == '.':
                    break
                file_name_limit += 1
            file_name_dict["filtered_inp_files/only_rem_duplicates/" + i] = i[0:file_name_limit]

        input_dict = get_nodes(input_dict, file_name_limit)
        text_info = get_text(input_dict)

        input_dict = get_chords(input_dict, file_name_dict)
        input_dict = color_chords(input_dict)  # changes colors for chords
        input_dict = filter_chords(input_dict)  # filters chords by removing duplicates

        for i in range(11):
            # the color pallete value parameter won't work unless you have variance, so this piece of code adds a tiny, tiny chord of every color which tricks the color pallete
            input_dict["Chords"].append({
                "source": {

                    "id": input_dict["Nodes"][0]['id'],
                    "start": 0,
                    "end": 0.000000005
                },
                "target": {
                    "id": input_dict["Nodes"][1]['id'],
                    "start": 0,
                    "end": 0.000000005
                },

                "value": float(i / 10)
            })

        with open('cache/input_d1.pkl', 'wb') as f:
            pickle.dump(input_dict, f)
    else:
        print("USING CACHED DATA, SINCE -f FLAG WAS MISSING...\n")

        with open('cache/input_d1.pkl', 'rb') as f:
            input_dict = pickle.load(f)

        text_info = get_text(input_dict)
    # app and callback configuration

    print("ADDING MAIN GRAPH")
    children_for_graph = []
    children_for_graph.append(dcc.Checklist(
        id="checklist",
        options=[
            {'label': '126', 'value': '126'},
            {'label': '253', 'value': '253'}
        ],
        value=[]
    ))

    graph = dashbio.Circos(
        id="hpc_circos_no_chord_data",
        selectEvent={"0": "hover"},
        layout=input_dict["Nodes"],
        config=layout_config,
        tracks=[
            {"type": "CHORDS", "data": input_dict["Chords"], "config": chords_config, 'id': "chords_no_chord_data", },
            {"type": "TEXT", "data": text_info, "config": text_config},
        ],
        enableDownloadSVG=True
    )

    children_for_graph.append(html.Div([

        html.H3('Graph 1: (REMOVED NODES WITHOUT CHORD DATA)'),
        graph

    ], style={'display': 'block'}

    ))
    children_for_graph.append(html.Div(["Event data for Graph 1:",
                                        html.Div(id="default-circos-output_no_chord_data")]))
    # finished adding 2 main graphs
    # now need to add other individual graphs

    print("ADDING INDEPENDENT GRAPHS")

    for i in range(len(os.listdir("filtered_inp_files/only_rem_duplicates/")) - 1):
        new_dict = {"Nodes": [], "Chords": []}
        for node in input_dict["Nodes"]:
            new_dict["Nodes"].append(node)

        for chord in input_dict["Chords"]:
            if chord["source"]["end"] != 0.000000005:
                new_dict["Chords"].append(chord)
        new_dict = filter_nodes(new_dict, i + 1)
        print("For " + str(i + 1) + " connections, nodes: " + str(new_dict["Nodes"]))
        if len(new_dict["Nodes"]) != 0:
            for j in range(11):
                # the color pallete value parameter won't work unless you have variance, so this piece of code adds a tiny, tiny chord of every color which tricks the color pallete
                new_dict["Chords"].append({
                    "source": {

                        "id": new_dict["Nodes"][0]['id'],
                        "start": 0,
                        "end": 0.000000005  # this represents a chord that was added for color
                    },
                    "target": {
                        "id": new_dict["Nodes"][1]['id'],
                        "start": 0,
                        "end": 0.000000005
                    },

                    "value": float(j / 10)
                })
        new_text_info = get_text(new_dict)
        graph = dashbio.Circos(
            id="hpc_circos_" + str(i + 1),
            selectEvent={"0": "hover"},
            layout=new_dict["Nodes"],
            config=layout_config,
            tracks=[
                {"type": "CHORDS", "data": new_dict["Chords"], "config": chords_config, 'id': "chords_" + str(i + 1), },
                {"type": "TEXT", "data": new_text_info, "config": text_config},
            ],
            enableDownloadSVG=True
        )

        children_for_graph.append(html.Div([

            html.H3('Graph showing nodes with ' + str(i + 1) + ' connection(s) (REMOVED NODES WITHOUT CHORD DATA)'),
            graph

        ], style={'display': 'block'},

        ),
        )
        children_for_graph.append(html.Div(["Event data for Graph " + str(i + 1) + ":",
                                            html.Div(id="default-circos-output_" + str(i + 1))]))

    app.layout = html.Div(children=children_for_graph)


    @callback(
        Output(component_id='default-circos-output_no_chord_data', component_property='children'),
        Input("hpc_circos_no_chord_data", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][
                                                                       :len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + \
                             value["target"]["id"] + " (of severity: " + \
                             str(value["target_severity"]) + ")"
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."


    @callback(
        Output(component_id='default-circos-output_1', component_property='children'),
        Input("hpc_circos_1", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][
                                                                       :len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + \
                             value["target"]["id"] + " (of severity: " + \
                             str(value["target_severity"]) + ")"
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."


    @callback(
        Output(component_id='default-circos-output_2', component_property='children'),
        Input("hpc_circos_2", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][
                                                                       :len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + \
                             value["target"]["id"] + " (of severity: " + \
                             str(value["target_severity"]) + ")"
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."


    @callback(
        Output(component_id='default-circos-output_3', component_property='children'),
        Input("hpc_circos_3", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][
                                                                       :len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + \
                             value["target"]["id"] + " (of severity: " + \
                             str(value["target_severity"]) + ")"
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."


    app.run_server()