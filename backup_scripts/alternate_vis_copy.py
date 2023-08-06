import random

import pandas as pd
import os
import statistics
import json
import urllib.request as urlreq
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bio as dashbio
import yaml
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import math
import sys
import numpy as np
import plotly.express as px

app = Dash(__name__)

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

chords_config = {"color": "RdYlBu",
                 "opacity": 0.8,
                 'tooltipContent': {
                     'source': 'source',
                     'sourceID': 'id',
                     'target': 'target',
                     'targetID': 'id',
                     'targetEnd': 'end'},
                 }
"""'axes': [
    {
      'color': 'red',
      'position': 4,
      'opacity': 0.3
    },
    {
      'color': 'red',
      'spacing': 2,
      'end': 4
    },
    {
      'color': 'green',
      'spacing': 1,
      'start': 4,
      'end': 16,
      'thickness': 1
    }
  ]"""




# RdYlBu is essentially the color scale (from https://github.com/d3/d3-scale-chromatic) that we are configuring the chords to use. (This is what the 'value' parameter represents)

def min_max_scaler(data, feature_range):
    # standard min-max scaling
    if len(data) == 0:
        return []
    scaler = MinMaxScaler(feature_range=feature_range)
    # transform data
    scaled = scaler.fit_transform(data)
    return scaled


def min_max_scaler_sum(data, limit):
    # min max scaling which scales the numbers in such a way that the sum of the numbers can be user specified (limit)
    if len(data) == 0:
        return []
        # if length of data is 0, then that means that there is no function with the given severity, which means that there should be no chord data drawn from this node
    normalizer = limit / float(sum(data))

    # multiply each item by the normalizer
    numListNormalized = [x * normalizer for x in data]
    return numListNormalized

def robust_scaler(data, limit):
    median = statistics.median(data)
    q3, q1 = np.percentile(data, [75, 25])
    iqr = q3 - q1
    for i in range(len(data)):
        data[i]=(data[i]-median)/iqr
        data[i]=data[i]*limit
    return data



def get_text(input_dict):
    # configures the text
    # mainly makes sure that the text goes OUTSIDE of the nodes

    text_info = []
    for node in input_dict['Nodes']:
        text_info.append(
            {
                "block_id": node['id'],
                "position": node['len'] / 2,
                "value": node['id']#[:len(node['id'])-2]

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
    # references our input dictionary to return the length of the specified node
    for node in input_dict['Nodes']:
        if node['id'] == node_id:
            return node['len']
    return 0

def reverse(lst):
   new_lst = lst[::-1]
   return new_lst

def get_nodes_rmve_empty(input_dict, color_list, file_name_limit):

    print("Building Nodes...\n")
    """
    id:
    label:
    color:
    len:
    """
    # get dictionary that represents number of calls of certain call stack

    # will retain order with corresponding call_stack
    color_list = ["#232023", "#A7A6BA", "#808080", "#C5C6D0", "#D3D3D3"]
    index=0
    for i in os.listdir("filtered_inp_files/only_rem_duplicates/"):
        original_df = pd.read_csv(
            "filtered_inp_files/only_rem_duplicates/" + str(os.listdir("filtered_inp_files/only_rem_duplicates/")[index]))
        combined_func_df = pd.read_csv("filtered_inp_files/func_data/combined_func_data.csv")
        n_calls_dict = {}

        for call_stack in combined_func_df['call_stack']:
            total_calls = 0
            for j in range(len(original_df['call_stack'])):
                if call_stack == original_df['call_stack'][j]:
                    total_calls += original_df['num_calls'][j]
            n_calls_dict[call_stack] = total_calls


        #print(n_calls_normalized) #may not be the best way to normalize



        if index%2==0:
            for call_stack in combined_func_df['call_stack']:
                if n_calls_dict[call_stack]!=0:
                    input_dict['Nodes'].append({
                        "id" : str(call_stack)+"_"+i[0:file_name_limit],
                        "label" : str(call_stack)+"_"+i[0:file_name_limit],
                        "color" : color_list[index],
                        "len" : n_calls_dict[call_stack]

                    })
        else:

            for call_stack in reverse(combined_func_df['call_stack'].tolist()):
                if n_calls_dict[call_stack] != 0:
                    input_dict['Nodes'].append({
                        "id": str(call_stack) + "_" + i[0:file_name_limit],
                        "label": str(call_stack) + "_" + i[0:file_name_limit],
                        "color": color_list[index],
                        "len": n_calls_dict[call_stack],
                        "opacity":0.3

                    })
        index+=1
    return input_dict

def get_nodes_rmve_none(input_dict, color_list, file_name_limit):

    print("Building Nodes...\n")
    """
    id:
    label:
    color:
    len:
    """
    # get dictionary that represents number of calls of certain call stack

    # will retain order with corresponding call_stack

    color_list = ["#232023", "#A7A6BA", "#808080", "#C5C6D0", "#D3D3D3"]
    index=0
    for i in os.listdir("filtered_inp_files/only_rem_duplicates/"):
        original_df = pd.read_csv(
            "filtered_inp_files/only_rem_duplicates/" + str(os.listdir("filtered_inp_files/only_rem_duplicates/")[index]))
        combined_func_df = pd.read_csv("filtered_inp_files/func_data/combined_func_data.csv")
        n_calls_dict = {}

        for call_stack in combined_func_df['call_stack']:
            total_calls = 0
            for j in range(len(original_df['call_stack'])):
                if call_stack == original_df['call_stack'][j]:
                    total_calls += original_df['num_calls'][j]
            n_calls_dict[call_stack] = total_calls


        #print(n_calls_normalized) #may not be the best way to normalize



        if index%2==0:
            for call_stack in combined_func_df['call_stack']:
                if n_calls_dict[call_stack]!=0:
                    input_dict['Nodes'].append({
                        "id" : str(call_stack)+"_"+i[0:file_name_limit],
                        "label" : str(call_stack)+"_"+i[0:file_name_limit],
                        "color" : color_list[index],
                        "len" : n_calls_dict[call_stack]

                    })
                else:
                    input_dict['Nodes'].append({
                        "id": str(call_stack) + "_" + i[0:file_name_limit],
                        "label": str(call_stack) + "_" + i[0:file_name_limit],
                        "color": "#FF0000",
                        "len": 20,
                        "opacity": 0.3
                    })
        else:

            for call_stack in reverse(combined_func_df['call_stack'].tolist()):
                if n_calls_dict[call_stack] != 0:
                    input_dict['Nodes'].append({
                        "id": str(call_stack) + "_" + i[0:file_name_limit],
                        "label": str(call_stack) + "_" + i[0:file_name_limit],
                        "color": color_list[index],
                        "len": n_calls_dict[call_stack],
                        "opacity":0.3

                    })
                else:
                    input_dict['Nodes'].append({
                        "id": str(call_stack) + "_" + i[0:file_name_limit],
                        "label": str(call_stack) + "_" + i[0:file_name_limit],
                        "color": "#FF0000",
                        "len": 20,
                        "opacity": 0.3
                    })
        index+=1
    return input_dict


def calc_limit(x):
    # through manual testing, this function was found to provide an effective relation between the chord spacing factor and the top number of functions (user specified)
    return math.ceil(1082.06 * (pow(math.e, 0.000435115 * x)) - 1076.51)



def get_chords(input_dict, file_name_dict):
    for json_dump_index in range(len(os.listdir("filtered_inp_files/only_rem_duplicates/"))):
        # go through chord data folder that is categorized by JSON Dump

        print("Going through " + str(os.listdir("filtered_inp_files/only_rem_duplicates/")[json_dump_index]))





        # get the JSON dumps to compare the previous one with
        json_dumps_to_compare_with = os.listdir("filtered_inp_files/only_rem_duplicates/")
        del json_dumps_to_compare_with[json_dump_index]

        original_df = pd.read_csv(
            "filtered_inp_files/only_rem_duplicates/" + str(os.listdir("filtered_inp_files/only_rem_duplicates/")[json_dump_index]))

       # scaled = min_max_scaler(original_df['exclusive_runtimes'].to_numpy().reshape(-1, 1),
        #                        (0, source_node_length / calc_limit(top_number_func)))
       # df_by_severity['scaled_runtimes'] = scaled



        for i in range(len(original_df)):
            source_start = 0
            #value_list_2=[0.5, 1]

            n=8
            total_possible_connections = len(json_dumps_to_compare_with)# * len(original_df['call_stack'].tolist())/n
            num_chords_from_specific_node = 0


            for j in range(len(json_dumps_to_compare_with)):



                # loop is used to compare our top functions (specified by user input) with the other JSON Dumps to find matches

                # target start indicates where on the target node the chord will "start"

                # create and configure a dataframe that contains the JSON dump that we are comparing our df_by_severity to
                comparing_to_df = pd.read_csv("filtered_inp_files/only_rem_duplicates/" + str(json_dumps_to_compare_with[j]))

                #scale runtimes
                original_df["scaled_runtimes"]=min_max_scaler(np.reshape(original_df["exclusive_runtimes"].tolist(), (-1, 1)), (0.1, 1))
                comparing_to_df["scaled_runtimes"] = min_max_scaler(
                    np.reshape(comparing_to_df["exclusive_runtimes"].tolist(), (-1, 1)), (0.1, 1))


                # go through df_by_severity to compare and find similarities in other datasets

                # print("Comparing with Dump " + str(json_dumps_to_compare_with[j][0:file_name_limit]))

                for k in range(len(comparing_to_df)):
                    # go through comparing_to_df to compare with df_by_severity

                    if (original_df['call_stack'].tolist()[i] ==
                            comparing_to_df['call_stack'].tolist()[k]):
                        num_chords_from_specific_node+=1

                        # check if there is a connection: one function in our source dataframe/JSON dump is found in another dump
                        # this indicates that a chord needs to be drawn

                        """if str(comparing_to_df['severity'].tolist()[k]) != previous_severity:
                            target_start = 0"""
                            # have target_start be 0, which means that it will start at the beginning of the given node

                        # create the chord based on this connection

                        source_id = original_df['call_stack'].tolist()[i]+"_"+file_name_dict["filtered_inp_files/only_rem_duplicates/"+str(os.listdir("filtered_inp_files/only_rem_duplicates/")[json_dump_index])]
                        target_id = comparing_to_df['call_stack'].tolist()[k]+"_"+file_name_dict["filtered_inp_files/only_rem_duplicates/"+json_dumps_to_compare_with[j]]

                        #source_increment = (get_length_of_node(source_id, input_dict)*5000000)/(total_possible_connections*original_df['exclusive_runtimes'].tolist()[i])
                        #source_increment = (get_length_of_node(source_id, input_dict)) / (
                        #            total_possible_connections * original_df['scaled_runtimes'].tolist()[i])

                        #target_increment = (get_length_of_node(target_id, input_dict) * 5000000) / (total_possible_connections * comparing_to_df['exclusive_runtimes'].tolist()[k])
                        #target_increment = (get_length_of_node(target_id, input_dict)) / (
                        #            total_possible_connections * comparing_to_df['scaled_runtimes'].tolist()[k])

                        """
                        we know the increment has to be this, regardless of the runtime because what we are doing is : (nodewidth*runtime_1)/(runtime_1+runtime_2) where runtime_1=runtime_2=runtime_3
                        This comes out to node_width/2, and 2 is the number of total possible connections
                        """
                        source_increment = 0.75*get_length_of_node(source_id, input_dict) / (total_possible_connections)

                        target_increment = 0.75*get_length_of_node(target_id, input_dict) / (total_possible_connections)

                        """scale_factor = 1
                        while True:
                            if ((source_increment+5)*(scale_factor+1)<get_length_of_node(source_id, input_dict)) and ((target_increment+5)*(scale_factor+1)<get_length_of_node(target_id, input_dict)):
                                scale_factor+=1

                        source_increment *= scale_factor
                        target_increment *= scale_factor"""
                        target_start=get_length_of_node(target_id, input_dict)-target_increment

                        #print("Call stack: " + source_id + " going to " +target_id)
                        #print(num_chords_from_specific_node)


                        input_dict["Chords"].append({
                            "source": {

                                "id": source_id,
                                "start": source_start,
                                "end": source_start+source_increment
                            },
                            "target": {
                                "id": target_id,
                                "start": target_start,
                                "end": target_start+target_increment
                            },
                            #"value": 0.5 ,
                            #value_list[num_chords_from_specific_node-1], #bug is that it counts when things that have 2 connections are at 1 connection
                            "call_stack":source_id,
                            "source_severity": original_df['sev_bin_upper_bound'][i],
                            "target_severity": comparing_to_df['sev_bin_upper_bound'][k]

                        })
                        source_start+=source_increment
                        source_start+=(get_length_of_node(source_id, input_dict) / (total_possible_connections))-(0.75*get_length_of_node(source_id, input_dict) / (total_possible_connections))

                        target_start-=target_increment
                        target_start+=(get_length_of_node(target_id, input_dict) / (total_possible_connections))-(0.75*get_length_of_node(target_id, input_dict) / (total_possible_connections))
    return input_dict

def reevaluate_nodes(input_dict):
    #remove nodes without any connections
    all_node_id =[]
    for dict in input_dict["Nodes"]:
        all_node_id.append(dict['id'])


    for dict in input_dict["Chords"]:
        if (dict["source"]["id"] in all_node_id):
            all_node_id.remove(dict["source"]["id"])
        if dict["target"]["id"] in all_node_id:
            all_node_id.remove(dict["target"]["id"])
    #now we have the ones that have no connections
    index =0

    for i in range(len(input_dict["Nodes"])):
        if input_dict["Nodes"][index]['id'] in all_node_id:
            #print(all_node_id)
            #print(input_dict["Nodes"][index]['id'])


            all_node_id.remove(input_dict["Nodes"][index]['id'])
            del input_dict["Nodes"][index]
            index-=1
        index+=1
    return input_dict

def reevaluate_chords(input_dict):
    #assign color values based on number of connections for graphs 1 and 2 (individual, independent graphs will be evaluated in the filter_nodes function)
    value_list = [1, 0.5, 0]
    all_node_id = []
    for dict in input_dict["Nodes"]:
        all_node_id.append(dict['id'])

    for id in all_node_id:
        counter_for_func = 0
        for dict in input_dict["Chords"]:
            if dict["source"]["id"]==id:
                counter_for_func+=1
        #print(id  + ": " + str(counter_for_func))
        for dict in input_dict["Chords"]:
            if dict["source"]["id"] == id:
                dict["value"] = value_list[counter_for_func-1]

    return input_dict

def check_num_connections(node_id, input_dict):
    num_connections=0
    for chord in input_dict["Chords"]:
        if chord["source"]["id"]==node_id:
            num_connections+=1
    return num_connections



def filter_nodes(original_dict, num_connections):
    #value_list = [0, 0.5, 1.0]
    value_list = [1, 0.5, 0] #will need to be extended if more than 4 dumps are being compared
    #filters nodes by number of connections
    nodes_to_delete=[]
    for node in original_dict["Nodes"]:
        if num_connections != check_num_connections(node["id"], original_dict):
            #print(str(node["id"]) + " has " + str(check_num_connections(node["id"], original_dict)) + " connections")
            nodes_to_delete.append(node["id"])

    index = 0

    for i in range(len(original_dict["Nodes"])):
        if original_dict["Nodes"][index]['id'] in nodes_to_delete:

            #nodes_to_delete.remove(input_dict["Nodes"][index]['id'])
            del original_dict["Nodes"][index]
            index -= 1
        index += 1

    index=0
    print(str(num_connections) + " connections corresponds to: " + str(value_list[num_connections - 1]))

    #try:
    if len(original_dict["Nodes"]) != 0:

        """except:
            #this will come when there are no nodes that only have num_connections number of connections
            pass"""

        for i in range(len(original_dict["Chords"])):
            if original_dict["Chords"][index]["source"]["id"] in nodes_to_delete:
                del original_dict["Chords"][index]
                index-=1
            else:
                original_dict["Chords"][index]["value"]=value_list[num_connections-1]
            index+=1
        for i in range(11):
            #the color pallete value parameter won't work unless you have variance, so this piece of code adds a tiny, tiny chord of every color which tricks the color pallete
            original_dict["Chords"].append({
                    "source": {

                        "id": original_dict["Nodes"][0]['id'],
                        "start": 0,
                        "end": 0.00000005
                    },
                    "target": {
                        "id": original_dict["Nodes"][1]['id'],
                        "start": 0,
                        "end": 0.000000005
                    },

                    "value" : float(i/10)
            })

        #add a chord with color white to make the other chords have color (otherwise it will have all black)
    return original_dict






if __name__ == '__main__':

    """

        Command Line Params:

        python vis.py top_num_func file_delim

        top_num_func (optional): a number that represents the top number of functions to display from each severity bin

        file_delim (optional) : file delimeter to accurately interpret files (will recognize '30' in filename 30_func_and_runtime.csv)



    """

    # gathering and prepping all dataframes
    argv = sys.argv
    del argv[0]
    if "-f" in argv:

        input_dict = {"Nodes": [], "Chords": []}
        """
        Blue: "#797EF6"
        Orange: "#FF781F"
        Red: "#FF781F"
        """
        #color_list = ["#797EF6", "#FF781F", "#B30000","#AFE1AF"]
        color_list = ["#FF7417", "#FB607F", "777B7E", "#708238"] #no Red, Blue, or Yellow to conflict with chords

        file_name_limit=3
        input_dict = get_nodes_rmve_empty(input_dict, color_list, file_name_limit)
        text_info = get_text(input_dict)

        file_name_dict={}
        for i in os.listdir("filtered_inp_files/only_rem_duplicates/"):
            file_name_dict["filtered_inp_files/only_rem_duplicates/"+i]=i[0:file_name_limit]

        input_dict = get_chords(input_dict, file_name_dict)
        input_dict = reevaluate_chords(input_dict)


        input_dict_2={"Nodes": [], "Chords": []}
        input_dict_2=get_nodes_rmve_none(input_dict_2, color_list, file_name_limit)

        for chord in input_dict["Chords"]:
            input_dict_2["Chords"].append(chord)

        for i in range(11):
            # the color pallete value parameter won't work unless you have variance, so this piece of code adds a tiny, tiny chord of every color which tricks the color pallete
            input_dict["Chords"].append({
                "source": {

                    "id": input_dict["Nodes"][0]['id'],
                    "start": 0,
                    "end": 0.00000005
                },
                "target": {
                    "id": input_dict["Nodes"][1]['id'],
                    "start": 0,
                    "end": 0.000000005
                },

                "value": float(i / 10)
            })
        #input_dict_2 = get_chords(input_dict_2, file_name_dict, value_list)
        input_dict_2 = reevaluate_chords(input_dict_2)

        for i in range(11):
            # the color pallete value parameter won't work unless you have variance, so this piece of code adds a tiny, tiny chord of every color which tricks the color pallete
            input_dict_2["Chords"].append({
                "source": {

                    "id": input_dict["Nodes"][0]['id'],
                    "start": 0,
                    "end": 0.00000005
                },
                "target": {
                    "id": input_dict["Nodes"][1]['id'],
                    "start": 0,
                    "end": 0.000000005
                },

                "value": float(i / 10)
            })
        text_info_2 = get_text(input_dict_2)

        with open('cache/input_d1.pkl', 'wb') as f:
            pickle.dump(input_dict, f)
        with open('cache/input_d2.pkl', 'wb') as f:
            pickle.dump(input_dict_2, f)

    else:
        with open('cache/input_d1.pkl', 'rb') as f:
            input_dict = pickle.load(f)
        with open('cache/input_d2.pkl', 'rb') as f:
            input_dict_2 = pickle.load(f)
        text_info = get_text(input_dict)
        text_info_2 = get_text(input_dict_2)

    children_for_graph = []

    graph = dashbio.Circos(
        id="hpc_circos_no_chord_data",
        selectEvent={"0": "hover"},
        layout=input_dict["Nodes"],
        config=layout_config,
        tracks=[
            {"type": "CHORDS", "data": input_dict["Chords"], "config": chords_config, 'id': "chords", },
            {"type": "TEXT", "data": text_info, "config": text_config},
        ],
        enableDownloadSVG=True
    )
    # graph.write_image("out/fig_no_chord_data.png")


    children_for_graph.append(html.Div([

        html.H3('Graph 1: (REMOVED NODES WITHOUT CHORD DATA)'),
        graph

    ], style={'display': 'block'}

    ))
    children_for_graph.append(html.Div(["Event data for Graph 1:",
                                        html.Div(id="default-circos-output_no_chord_data")]))

    graph = dashbio.Circos(
        id="hpc_circos_with_chord_data",
        selectEvent={"0": "hover"},
        layout=input_dict_2["Nodes"],
        config=layout_config,
        tracks=[
            {"type": "CHORDS", "data": input_dict_2["Chords"], "config": chords_config, 'id': "chords", },
            {"type": "TEXT", "data": text_info_2, "config": text_config},

        ],
        enableDownloadSVG=True
    )
    # graph.write_image("out/fig_with_chord_data.png")

    children_for_graph.append(html.Div([
        html.H3('Graph 2'),
        graph
        ,
    ], style={'display': 'block'}
    ))

    children_for_graph.append(html.Div(["Event data for Graph 2:",
                                        html.Div(id="default-circos-output_with_chord_data")]))
    # finished adding 2 main graphs
    # now need to add other individual graphs

    for i in range(len(os.listdir("filtered_inp_files/only_rem_duplicates/")) - 1):
        new_dict = {"Nodes": [], "Chords": []}
        for node in input_dict["Nodes"]:
            new_dict["Nodes"].append(node)

        for chord in input_dict["Chords"]:
            new_dict["Chords"].append(chord)
        new_dict = filter_nodes(new_dict, i + 1)
        # print(new_dict)
        new_text_info = get_text(new_dict)
        graph = dashbio.Circos(
            id="hpc_circos_" + str(i + 1),
            selectEvent={"0": "hover"},
            layout=new_dict["Nodes"],
            config=layout_config,
            tracks=[
                {"type": "CHORDS", "data": new_dict["Chords"], "config": chords_config, 'id': "chords", },
                {"type": "TEXT", "data": new_text_info, "config": text_config},
            ],
            enableDownloadSVG=True
        )
        # graph.write_image("out/fig_"+str(i+1)+"_connections.png")
        children_for_graph.append(html.Div([

            html.H3('Graph showing nodes with ' + str(i + 1) + ' connection(s) (REMOVED NODES WITHOUT CHORD DATA)'),
            graph

        ], style={'display': 'block'},

        ),
        )
        children_for_graph.append(html.Div(["Event data for Graph " + str(i + 1) + ":",
                                            html.Div(id="default-circos-output_" + str(i + 1))]))

    app.layout = html.Div(children=children_for_graph)
    app.run_server()





    @callback(
        Output(component_id='default-circos-output_with_chord_data', component_property='children'),
        Input("hpc_circos_with_chord_data", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][:len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + \
                             value["target"]["id"] + "(of severity: " + \
                             str(value["target_severity"]) + "). Also this chord has value parameter: " + str(value["value"])
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."


    @callback(
        Output(component_id='default-circos-output_no_chord_data', component_property='children'),
        Input("hpc_circos_no_chord_data", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][:len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + value["target"]["id"] + "(of severity: " + \
                             str(value["target_severity"]) + "). Also this chord has value parameter: " + str(value["value"])
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
                             value["target"]["id"] + "(of severity: " + \
                             str(value["target_severity"]) + "). Also this chord has value parameter: " + str(value["value"])
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
                             value["target"]["id"] + "(of severity: " + \
                             str(value["target_severity"]) + "). Also this chord has value parameter: " + str(value["value"])
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
                             value["target"]["id"] + "(of severity: " + \
                             str(value["target_severity"]) + "). Also this chord has value parameter: " + str(value["value"])
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."