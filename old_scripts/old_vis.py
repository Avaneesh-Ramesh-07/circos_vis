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


app=Dash(__name__)

layout_config = {

"labels": {"display": False},
    "ticks": {"display": False},
}

text_config = {
    "innerRadius": 1.02,
    "style": {
        "font-size": 12,
    },
}

chords_config = {"color": "RdYlBu", "opacity": 0.8}
#RdYlBu is essentially the color scale (from https://github.com/d3/d3-scale-chromatic) that we are configuring the chords to use. (This is what the 'value' parameter represents)

def min_max_scaler(data, feature_range):
    #standard min-max scaling
    if len(data)==0:
        return []
    scaler = MinMaxScaler(feature_range=feature_range)
    # transform data
    scaled = scaler.fit_transform(data)
    return scaled

def min_max_scaler_sum(data, limit):
    #min max scaling which scales the numbers in such a way that the sum of the numbers can be user specified (limit)
    if len(data)==0:
        return []
        #if length of data is 0, then that means that there is no function with the given severity, which means that there should be no chord data drawn from this node
    normalizer = limit / float(sum(data))

    # multiply each item by the normalizer
    numListNormalized = [x * normalizer for x in data]
    return numListNormalized

def get_text(input_dict):
    #configures the text
    #mainly makes sure that the text goes OUTSIDE of the nodes

    text_info=[]
    for node in input_dict['Nodes']:
        text_info.append(
            {
                "block_id": node['id'],
                "position": node['len']/2,
                "value": node['id']

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
    #references our input dictionary to return the length of the specified node
    for node in input_dict['Nodes']:
        if node['id']==node_id:
            return node['len']

def get_nodes(df_list, input_dict, color_list, file_name_list):
    index = 0
    #go through each dataframe, calculate the total runtime, and configure the dictionary so that it will properly graph the chords
    for dataframe in df_list:
        total_runtime = dataframe.iloc[0]["TotalRuntimeSum(AllFunc)"]
        for i in range(11):
            if (int(dataframe["agg_runtimes"][i])/total_runtime)!=0:
                input_dict["Nodes"].append({
                    "id": str(file_name_list[index])+"." + str(float(i/10)),
                    "label": str(file_name_list[index]) + ": " + str(float(i / 10)),
                    "color": color_list[index],
                    "len": int(dataframe["agg_runtimes"][i])/total_runtime
                })
        index += 1
    return input_dict

def calc_limit(x):
    #through manual testing, this function was found to provide an effective relation between the chord spacing factor and the top number of functions (user specified)
    return math.ceil(1082.06*(pow(math.e, 0.000435115*x))-1076.51)

def get_chords(input_dict, file_name_list, value_list, file_name_limit, top_number_func):

    severity_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for json_dump_index in range(len(os.listdir("filtered_inp_files/chord_data/"))):
        #go through chord data folder that is categorized by JSON Dump


        #get the JSON dumps to compare the previous one with
        json_dumps_to_compare_with=os.listdir("filtered_inp_files/chord_data/")
        del json_dumps_to_compare_with[json_dump_index]
        print("\nANALYZING DUMP " + file_name_list[json_dump_index]+"...")


        df=pd.read_csv("filtered_inp_files/chord_data/"+str(os.listdir("filtered_inp_files/chord_data/")[json_dump_index]))


        for severity in severity_bins:
            #go through each severity so we can generate the appropriate dataframe that contains every function in a certain dump for that specific severity
            severity_pos_in_list = severity_bins.index(severity)

            source_node_length = get_length_of_node(str(file_name_list[json_dump_index] + "." + str(severity_bins[severity_pos_in_list])), input_dict)
            if source_node_length is None: source_node_length=0
            #when source_node_length is None, this means that there isn't a source node, so the program shouldn't draw any chords from that
            #the program then assigns the source_node_length to be 0, and the df_by_severity to be a blank dataframe
            #our normalization function checks if the df_by_severity is a blank dataframe (meaning that there are no functions with that severity in the dump)
            #if df_by_severity is a blank dataframe, the normalization returns an empty list which later indicates that the df_by_severity will not be iterated through, so no chords are created and no errors are thrown


            #generate and configure dataframe that stores every possible function in that JSON dump under the specific severity
            #this will be used to find connections with the other dumps and ultimately graph the functions
            df_by_severity = pd.DataFrame(df[(df.severity == severity)])
            scaled = min_max_scaler(df_by_severity['total_runtimes'].to_numpy().reshape(-1, 1), (0, source_node_length/calc_limit(top_number_func)))
            df_by_severity['scaled_runtimes']=scaled
            #scaling runtimes so that you have the relevant "unit" to be able to graph the chord
            df_by_severity=df_by_severity.drop('Unnamed: 0', axis='columns')
            df_by_severity = df_by_severity.drop_duplicates()
            #print("in this dataset with severity: " + str(severity) + " there are: " + str(len(df_by_severity)) + " unique functions")
            df_by_severity=df_by_severity.nlargest(top_number_func, 'scaled_runtimes')


            source_start=0
            previous_severity=severity_bins[0]
            # previous severity variable is used to check whether the program has to move on to another severity bin
            # this will keep the program from graphing the chord to start from the wrong severity bin

            for j in range(len(json_dumps_to_compare_with)):
                #loop is used to compare our top functions (specified by user input) with the other JSON Dumps to find matches
                target_start=0
                #target start indicates where on the target node the chord will "start"


                #create and configure a dataframe that contains the JSON dump that we are comparing our df_by_severity to
                comparing_to_df = pd.read_csv("filtered_inp_files/chord_data/" + str(json_dumps_to_compare_with[j]))
                comparing_to_df = comparing_to_df.drop_duplicates(subset=['unique_call_stacks', 'severity'])
                comparing_to_df = comparing_to_df.drop('Unnamed: 0', axis='columns')


                for i in range(len(df_by_severity)):
                    #go through df_by_severity to compare and find similarities in other datasets

                    #print("Comparing with Dump " + str(json_dumps_to_compare_with[j][0:file_name_limit]))

                    for k in range(len(comparing_to_df)):
                        #go through comparing_to_df to compare with df_by_severity

                        if (df_by_severity['unique_call_stacks'].tolist()[i]==comparing_to_df['unique_call_stacks'].tolist()[k]):
                            #check if there is a connection: one function in our source dataframe/JSON dump is found in another dump
                            #this indicates that a chord needs to be drawn


                            if str(comparing_to_df['severity'].tolist()[k])!=previous_severity:
                                target_start=0
                                #have target_start be 0, which means that it will start at the beginning of the given node

                            #scaling and configuring comparing_to_df to include scaled runtimes. This will enable us to have a valid unit so we can plot the graph
                            target_node_length = get_length_of_node(str(json_dumps_to_compare_with[j][0:file_name_limit] + "." + str(comparing_to_df['severity'].tolist()[k])), input_dict)
                            if target_node_length is None: target_node_length = 0
                            scaled = min_max_scaler(comparing_to_df['total_runtimes'].to_numpy().reshape(-1, 1), (0, target_node_length / calc_limit(top_number_func)))
                            comparing_to_df['scaled_runtimes'] = scaled

                            #create the chord based on this connection
                            input_dict["Chords"].append({
                                "source": {

                                    "id": str(file_name_list[json_dump_index] + "." + str(severity_bins[severity_pos_in_list])),
                                    "start": source_start,
                                    "end": source_start+df_by_severity['scaled_runtimes'].tolist()[i]
                                },
                                "target": {
                                    "id": str(json_dumps_to_compare_with[j][0:file_name_limit] + "." + str(comparing_to_df['severity'].tolist()[k])),
                                    "start": target_start,
                                    "end": target_start+comparing_to_df['scaled_runtimes'].tolist()[k]
                                },
                                "value":value_list[json_dump_index],
                                "call_stack":df_by_severity['unique_call_stacks'].tolist()[i],
                                "target_start":target_start
                            })

                            previous_severity=str(comparing_to_df['severity'].tolist()[k])

                            #increment these so the next chord can start exactly at the end of the previous chord
                            source_start += df_by_severity['scaled_runtimes'].tolist()[i]
                            target_start+=comparing_to_df['scaled_runtimes'].tolist()[k]


    return input_dict

if __name__=='__main__':

    """

        Command Line Params:

        python vis.py top_num_func file_delim

        top_num_func (optional): a number that represents the top number of functions to display from each severity bin

        file_delim (optional) : file delimeter to accurately interpret files (will recognize '30' in filename 30_func_and_runtime.csv)



    """

    #gathering and prepping all dataframes

    argv = sys.argv
    del argv[0]
    if len(argv)==0:
        print("USING DEFAULT PARAMETERS SINCE NO COMMAND LINE INPUTS WERE FOUND\n")

        top_num_func = 5
        file_name_limit = 2
    elif len(argv)==1:
        print("USING DEFAULT VALUE FOR DELIMETER AS ONLY ONE VALUE WAS FOUND")
        top_num_func=int(argv[0])
        file_name_limit=2
    else:

        print("USING SPECIFIED PARAMETERS\n")
        try:
            top_num_func=int(argv[0])
            file_name_limit=int(argv[1])
        except:
            print("REVERTING TO DEFAULT PARAMETERS. PLEASE PROVIDE A NUMBER FOR ALL PARAMETERS.\n")
            top_num_func = 5
            file_name_limit = 2


    df_list = []

    #convert input csv files to dataframes
    for file_name in os.listdir("filtered_inp_files/unshortened/"):
        df_list.append(pd.read_csv("filtered_inp_files/unshortened/" + file_name))
    for dataframe in df_list:
        dataframe.set_index('call_stack', inplace=True)

    #file name list gives the file names to the get_chords function so it can match up the correct id when drawing the chord
    file_name_list=os.listdir("filtered_inp_files/unshortened/")
    for i in range(len(file_name_list)):
        file_name_list[i]=file_name_list[i][0:file_name_limit]

    #define input dict and fill it up
    input_dict={"Nodes":[], "Chords":[]}

    #assign color list in hex, and in 'value' which will be interpreted by value parameter in Dash Plotly Circos library
    color_list = ["#797EF6", "#FF781F", "#B30000"]
    value_list=[1,0.3,0.1]


    input_dict=get_nodes(df_list, input_dict, color_list, file_name_list)

    input_dict=get_chords(input_dict, file_name_list, value_list, file_name_limit, top_num_func)
    text_info=get_text(input_dict)


    #app and callback configuration
    app.layout = html.Div(
        [
        dashbio.Circos(
            id="hpc_circos",
            selectEvent={"0": "hover"},
            layout=input_dict["Nodes"],
            config=layout_config,
            tracks=[
                {"type": "CHORDS", "data": input_dict["Chords"], "config": chords_config, 'id': "chords"},
                {"type": "TEXT", "data": text_info, "config": text_config},

            ]
        ),
        "Event data:",
        html.Div(id="default-circos-output")
    ]
    )


    @callback(
        Output(component_id='default-circos-output', component_property='children'),
        Input("hpc_circos", "eventDatum"),
    )
    def update_output(value):

        if (value is not None):
            output_string=""
            output_string+="This chord, representing call stack, " + value["call_stack"] +", goes from " + value["source"]["id"] +" to " + value["target"]["id"] +". The sources and targets are formatted by: JSON_Dump.Severity."
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



