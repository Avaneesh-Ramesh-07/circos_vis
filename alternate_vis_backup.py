import pandas as pd
import os
import statistics
import json
import urllib.request as urlreq
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bio as dashbio

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import math
import sys
import numpy as np

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
                'axes': [
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
                  ]
                }



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

def get_nodes(input_dict, color_list, file_name_limit):

    print("Building Nodes...\n")
    """
    id:
    label:
    color:
    len:
    """
    # get dictionary that represents number of calls of certain call stack

    # will retain order with corresponding call_stack

    index=0
    for i in os.listdir("filtered_inp_files/unshortened/"):
        original_df = pd.read_csv(
            "filtered_inp_files/unshortened/" + str(os.listdir("filtered_inp_files/unshortened/")[index]))
        combined_func_df = pd.read_csv("filtered_inp_files/func_data/combined_func_data.csv")
        n_calls_dict = {}

        for call_stack in combined_func_df['call_stack']:
            total_calls = 0
            for j in range(len(original_df['call_stack'])):
                if call_stack == original_df['call_stack'][j]:
                    total_calls += original_df['num_calls'][j]
            n_calls_dict[call_stack] = total_calls


        #print(n_calls_normalized) #may not be the best way to normalize

        print(len(combined_func_df['call_stack'].tolist()))



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

    index=0
    for i in os.listdir("filtered_inp_files/unshortened/"):
        original_df = pd.read_csv(
            "filtered_inp_files/unshortened/" + str(os.listdir("filtered_inp_files/unshortened/")[index]))
        combined_func_df = pd.read_csv("filtered_inp_files/func_data/combined_func_data.csv")
        n_calls_dict = {}

        for call_stack in combined_func_df['call_stack']:
            total_calls = 0
            for j in range(len(original_df['call_stack'])):
                if call_stack == original_df['call_stack'][j]:
                    total_calls += original_df['num_calls'][j]
            n_calls_dict[call_stack] = total_calls


        #print(n_calls_normalized) #may not be the best way to normalize

        print(len(combined_func_df['call_stack'].tolist()))



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


def calc_limit(x):
    # through manual testing, this function was found to provide an effective relation between the chord spacing factor and the top number of functions (user specified)
    return math.ceil(1082.06 * (pow(math.e, 0.000435115 * x)) - 1076.51)



def get_chords(input_dict, file_name_dict, value_list):
    for json_dump_index in range(len(os.listdir("filtered_inp_files/unshortened/"))):
        # go through chord data folder that is categorized by JSON Dump

        print("Going through " + str(os.listdir("filtered_inp_files/unshortened/")[json_dump_index]))





        # get the JSON dumps to compare the previous one with
        json_dumps_to_compare_with = os.listdir("filtered_inp_files/unshortened/")
        del json_dumps_to_compare_with[json_dump_index]

        original_df = pd.read_csv(
            "filtered_inp_files/unshortened/" + str(os.listdir("filtered_inp_files/unshortened/")[json_dump_index]))

       # scaled = min_max_scaler(original_df['exclusive_runtimes'].to_numpy().reshape(-1, 1),
        #                        (0, source_node_length / calc_limit(top_number_func)))
       # df_by_severity['scaled_runtimes'] = scaled



        for i in range(len(original_df)):
            source_start = 0
            #value_list_2=[0.5, 1]

            n=8
            total_possible_connections = len(json_dumps_to_compare_with) * len(original_df['call_stack'].tolist())/n
            num_chords_from_specific_node = 0


            for j in range(len(json_dumps_to_compare_with)):



                # loop is used to compare our top functions (specified by user input) with the other JSON Dumps to find matches

                # target start indicates where on the target node the chord will "start"

                # create and configure a dataframe that contains the JSON dump that we are comparing our df_by_severity to
                comparing_to_df = pd.read_csv("filtered_inp_files/unshortened/" + str(json_dumps_to_compare_with[j]))


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

                        source_id = original_df['call_stack'].tolist()[i]+"_"+file_name_dict["filtered_inp_files/unshortened/"+str(os.listdir("filtered_inp_files/unshortened/")[json_dump_index])]
                        target_id = comparing_to_df['call_stack'].tolist()[k]+"_"+file_name_dict["filtered_inp_files/unshortened/"+json_dumps_to_compare_with[j]]

                        source_increment = (get_length_of_node(source_id, input_dict)*100000)/(total_possible_connections*original_df['exclusive_runtimes'].tolist()[i])
                        target_increment = (get_length_of_node(target_id, input_dict)*100000) / (total_possible_connections*comparing_to_df['exclusive_runtimes'].tolist()[k])

                        #source_increment = get_length_of_node(source_id, input_dict) / (total_possible_connections)
                        #target_increment = get_length_of_node(target_id, input_dict) / (total_possible_connections)


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
                            "value": 0.5 ,
                            #value_list[num_chords_from_specific_node-1], #bug is that it counts when things that have 2 connections are at 1 connection
                            "call_stack":source_id,
                            "source_severity": original_df['sev_bin_upper_bound'][i],
                            "target_severity": comparing_to_df['sev_bin_upper_bound'][k]

                        })
                        source_start+=source_increment
                        target_start-=target_increment
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
    value_list  = [0.1, 0.5, 0.7, 0.9]
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




if __name__ == '__main__':

    """

        Command Line Params:

        python vis.py top_num_func file_delim

        top_num_func (optional): a number that represents the top number of functions to display from each severity bin

        file_delim (optional) : file delimeter to accurately interpret files (will recognize '30' in filename 30_func_and_runtime.csv)



    """

    # gathering and prepping all dataframes

    """argv = sys.argv
    del argv[0]
    if len(argv) == 0:
        print("USING DEFAULT PARAMETERS SINCE NO COMMAND LINE INPUTS WERE FOUND\n")

        top_num_func = 5
        file_name_limit = 2
    elif len(argv) == 1:
        print("USING DEFAULT VALUE FOR DELIMETER AS ONLY ONE VALUE WAS FOUND")
        top_num_func = int(argv[0])
        file_name_limit = 2
    else:

        print("USING SPECIFIED PARAMETERS\n")
        try:
            top_num_func = int(argv[0])
            file_name_limit = int(argv[1])
        except:
            print("REVERTING TO DEFAULT PARAMETERS. PLEASE PROVIDE A NUMBER FOR ALL PARAMETERS.\n")
            top_num_func = 5
            file_name_limit = 2"""

    """df_list = []

    # convert input csv files to dataframes
    for file_name in os.listdir("filtered_inp_files/unshortened/"):
        df_list.append(pd.read_csv("filtered_inp_files/unshortened/" + file_name))
    for dataframe in df_list:
        dataframe.set_index('call_stack', inplace=True)

    # file name list gives the file names to the get_chords function so it can match up the correct id when drawing the chord
    file_name_list = os.listdir("filtered_inp_files/unshortened/")
    for i in range(len(file_name_list)):
        file_name_list[i] = file_name_list[i][0:file_name_limit]

    # define input dict and fill it up
    input_dict = {"Nodes": [], "Chords": []}

    # assign color list in hex, and in 'value' which will be interpreted by value parameter in Dash Plotly Circos library
    color_list = ["#797EF6", "#FF781F", "#B30000"]

    input_dict = get_nodes(df_list, input_dict, color_list, file_name_list)

    input_dict = get_chords(input_dict, file_name_list, value_list, file_name_limit, top_num_func)"""
    input_dict = {"Nodes": [], "Chords": []}
    """
    Blue: "#797EF6"
    Orange: "#FF781F"
    Red: "#FF781F"
    """
    #color_list = ["#797EF6", "#FF781F", "#B30000","#AFE1AF"]
    color_list = ["#FF7417", "#FB607F", "777B7E", "#708238"] #no Red, Blue, or Yellow to conflict with chords
    value_list = [0.1, 0.9, 0.1]
    file_name_limit=3
    input_dict = get_nodes(input_dict, color_list, file_name_limit)
    #print(input_dict)
    text_info = get_text(input_dict)

    file_name_dict={}
    for i in os.listdir("filtered_inp_files/unshortened/"):
        file_name_dict["filtered_inp_files/unshortened/"+i]=i[0:file_name_limit]
    #print(file_name_dict)

    input_dict = get_chords(input_dict, file_name_dict, value_list)

    input_dict_2={"Nodes": [], "Chords": []}

    for node in input_dict["Nodes"]:
        input_dict_2["Nodes"].append(node)
    for chord in input_dict["Chords"]:
        input_dict_2["Chords"].append(chord)
    input_dict_2=reevaluate_chords(input_dict_2)
    #print(input_dict["Nodes"])
    input_dict=reevaluate_nodes(input_dict)
    input_dict=reevaluate_chords(input_dict)
   # print(input_dict["Chords"])
    #print(input_dict["Nodes"])
    # app and callback configuration
    app.layout = html.Div(
        [
            dcc.Checklist(
                id='checklist',
                options=[
                    {'label': 'Graph 1', 'value': '1'},
                    {'label': 'Graph 2', 'value': '2'}
                ],
                value=['1','2']
            ),
            html.Div([

                html.H3('Graph 1: (REMOVED NODES WITHOUT CHORD DATA)'),
                dashbio.Circos(
                    id="hpc_circos",
                    selectEvent={"0": "hover"},
                    layout=input_dict["Nodes"],
                    config=layout_config,
                    tracks=[
                        {"type": "CHORDS", "data": input_dict["Chords"], "config": chords_config, 'id': "chords",},
                        {"type": "TEXT", "data": text_info, "config": text_config},
                        ],
                    )

            ], style= {'display': 'block'}
            ),


            "Event data for Graph 1:",
            html.Div(id="default-circos-output"),

            html.Div([
                html.H3('Graph 2'),

                dashbio.Circos(
                    id="hpc_circos_2",
                    selectEvent={"0": "hover"},
                    layout=input_dict_2["Nodes"],
                    config=layout_config,
                    tracks=[
                        {"type": "CHORDS", "data": input_dict_2["Chords"], "config": chords_config, 'id': "chords", },
                        {"type": "TEXT", "data": text_info, "config": text_config},

                    ],
                ),
            ], style= {'display': 'block'}
            ),

            "Event data for Graph 2:",
            html.Div(id="default-circos-output_2")
        ]
    )





    @callback(
        Output(component_id='default-circos-output', component_property='children'),
        Input("hpc_circos", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][:len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + \
                             value["target"]["id"] + "(of severity: " + \
                             str(value["target_severity"]) + ")."
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."


    @callback(
        Output(component_id='default-circos-output_2', component_property='children'),
        Input("hpc_circos_2", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string = ""
            output_string += "This chord, representing call stack, " + value["call_stack"][:len(value["call_stack"]) - 3] + ", goes from " + \
                             value["source"]["id"] + " (of severity:  " + str(value["source_severity"]) + ") to " + value["target"]["id"] + "(of severity: " + \
                             str(value["target_severity"]) + ")."
            return [html.Div(output_string)]
        return "There are no event data. Hover over a data point to get more information."


    app.run_server()


"""
OLD CODE DOWN HERE:
if (df_by_severity['total_runtimes'].tolist()==[]):
    scaled=[]
else:
    for unique_call_stack in df_by_severity.nlargest(2, 'total_runtimes')['unique_call_stacks']:
        num_dumps_found = 0
        for i in range(1, 3):

            comparing_to_df = pd.read_csv(
                "filtered_inp_files/chord_data/" + str(os.listdir("filtered_inp_files/chord_data/")[i]))

            comparing_to_df = comparing_to_df.drop_duplicates(subset=['unique_call_stacks', 'severity'])
            comparing_to_df = comparing_to_df.drop('Unnamed: 0', axis='columns')
            for k in range(len(comparing_to_df)):

                if (unique_call_stack == comparing_to_df['unique_call_stacks'].tolist()[k]):
                    num_dumps_found+=1
    if num_dumps_found<=len(df_by_severity.nlargest(2, 'total_runtimes')['unique_call_stacks']): #if the sev bin has only 1 element then it checks with 1, else it checks with 2
        scaled = min_max_scaler(df_by_severity['total_runtimes'].tolist(),
                                (node_length / (2)))
    else:
        scaled = min_max_scaler(df_by_severity['total_runtimes'].tolist(), (node_length / (2*(len(os.listdir("filtered_inp_files/chord_data/"))-1))))#if there are 2 other dumps, the maximum possibilities is that this one call stack can be found in both other dumps (2). We account for this by first dividing by 2 (to get half) and then dividing by the number of other dumps for this edge case


"""



