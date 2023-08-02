import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import statistics

##For reading json
import json
import plotly.graph_objects as go
from collections import OrderedDict
from dash_bootstrap_templates import load_figure_template

import hashlib
from collections import Counter
from collections import defaultdict
from datetime import datetime as dati
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
class DataFrame:
    def __init__(self):
        self.func_name = ""
        self.func_names_list = []
        self.exclusive_runtimes_list = []
        self.severity_list = []
        self.hashed_call_stack_list = []  # a hash value of the entire callstack
        self.hash_counts = {}
        self.exclusive_runtime = 0
        self.severity = 0
        self.norm_severity_list = []
        self.norm_exclusive_runtimes_list = []
        self.entry_time_list = []
        self.exit_time_list = []
        self.rank_list = []
        self.callstack_map = OrderedDict()
        self.callgraph_edgelist_per_function = OrderedDict()

    def return_first_entry_for_specific_callstack_hash(self, hashvalue):
        hashindex = self.hashed_call_stack_list.index(hashvalue)
        #         print("hashindex: ", hashindex)
        newobj = DataFrame()
        newobj.func_name = self.func_names_list[hashindex]
        newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
        newobj.severity = self.severity_list[hashindex]
        newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
        return newobj

    def return_allentries_for_specific_callstack_hash(self, hashvalue):
        dataframe_list = []
        prev = 0
        for start in range(len(self.hashed_call_stack_list)):
            try:
                hashindex = self.hashed_call_stack_list.index(
                    hashvalue, prev, len(self.hashed_call_stack_list)
                )
                newobj = DataFrame()
                newobj.func_name = self.func_names_list[hashindex]
                newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
                newobj.severity = self.severity_list[hashindex]
                newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
                dataframe_list.append(newobj)
                prev = (
                    hashindex + 1
                )  ##So that, the next position starts from hashindex+1
            except ValueError:
                #                 print("That item does not exist")
                break
        return dataframe_list

    def return_allentries_for_specific_severity(self, severity):
        dataframe_list = []
        prev = 0
        for start in range(len(self.severity_list)):
            try:
                hashindex = self.severity_list.index(
                    severity, prev, len(self.severity_list)
                )
                newobj = DataFrame()
                newobj.func_name = self.func_names_list[hashindex]
                newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
                newobj.severity = self.severity_list[hashindex]
                newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
                dataframe_list.append(newobj)
                prev = (
                    hashindex + 1
                )  ##So that, the next position starts from hashindex+1
            except ValueError:
                #                 print("That item does not exist")
                break
        return dataframe_list

    def return_allentries_for_specific_norm_severity(self, norm_severity):
        """
        This function returns a list of DataFrame objects that have greater than the specified normalized severity level.
        """
        dataframe_list = []
        # Get indices of all normalized severities that are greater than norm_severity
        res = [
            idx
            for idx, val in enumerate(self.norm_severity_list)
            if val >= norm_severity
        ]
        for hashindex in res:
            newobj = DataFrame()
            newobj.func_name = self.func_names_list[hashindex]
            newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
            newobj.severity = self.severity_list[hashindex]
            newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
            dataframe_list.append(newobj)
            prev = hashindex + 1  ##So that, the next position starts from hashindex+1
        return dataframe_list

    def return_unique_entries_for_specific_norm_severity(self, norm_severity):
        """
        This function returns a list of DataFrame objects with unique call_path hash keys that have greater than the specified normalized severity level.
        """
        dataframe_list = []
        # Get indices of all normalized severities that are greater than norm_severity
        res = [
            idx
            for idx, val in enumerate(self.norm_severity_list)
            if val >= norm_severity
        ]
        #print(len(res))
        unique_func_names = [self.func_names_list[i] for i in res]
        #print(set(unique_func_names))
        unique_callpath_hash = [self.hashed_call_stack_list[i] for i in res]
        #print(set(unique_callpath_hash))

    def return_unique_entries_for_specific_norm_severity(
        self, exclusive_lower_bound, inclusive_upper_bound
    ):
        dataframe_list = []
        # Get indices of all normalized severities that are greater than norm_severity
        res = [
            idx
            for idx, val in enumerate(self.norm_severity_list)
            if (val > exclusive_lower_bound) and (val <= inclusive_upper_bound)
        ]
        #print(len(res))
        unique_func_names = [self.func_names_list[i] for i in res]
        #print(set(unique_func_names))
        unique_callpath_hash = [self.hashed_call_stack_list[i] for i in res]
        #print(set(unique_callpath_hash))

        for hashindex in res:
            newobj = DataFrame()
            newobj.func_name = self.func_names_list[hashindex]
            newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
            newobj.severity = self.severity_list[hashindex]
            newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
            dataframe_list.append(newobj)
            prev = hashindex + 1  ##So that, the next position starts from hashindex+1
        return set(unique_func_names), set(unique_callpath_hash), dataframe_list

    def return_perfunc_entries_for_specific_norm_severity(
        self, exclusive_lower_bound, inclusive_upper_bound
    ):
        dataframe_list = []
        # Get indices of all normalized severities that are greater than norm_severity
        res = [
            idx
            for idx, val in enumerate(self.norm_severity_list)
            if (val > exclusive_lower_bound) and (val <= inclusive_upper_bound)
        ]
        #         print('Number of entries for severity: (', exclusive_lower_bound, ",", inclusive_upper_bound, "]", len(res))
        unique_func_names = [self.func_names_list[i] for i in res]
        unique_func_names = set(unique_func_names)
        func_dict = tree()
        for func_name in unique_func_names:
            res = [
                idx
                for idx, val in enumerate(self.norm_severity_list)
                if (val > exclusive_lower_bound)
                and (val <= inclusive_upper_bound)
                and self.func_names_list[idx] == func_name
            ]
            #             tmp_list = [self.hashed_call_stack_list[i] for i in res]
            unique_callpath_hash = [self.hashed_call_stack_list[i] for i in res]
            unique_callpath = set(unique_callpath_hash)

            dataframe_list = []
            for cp in unique_callpath:
                hashindex_list = [
                    idx
                    for idx, val in enumerate(self.norm_severity_list)
                    if (val > exclusive_lower_bound)
                    and (val <= inclusive_upper_bound)
                    and self.func_names_list[idx] == func_name
                    and (self.hashed_call_stack_list[idx] == cp)
                ]
                for hashindex in hashindex_list:
                    newobj = DataFrame()
                    newobj.func_name = func_name
                    newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
                    newobj.severity = self.severity_list[hashindex]
                    newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
                    dataframe_list.append(newobj)

            #             for hashindex in res:
            #                 if self.func_names_list[hashindex] == func_name
            #                 newobj = DataFrame()
            #                 newobj.func_name = func_name
            #                 newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
            #                 newobj.severity = self.severity_list[hashindex]
            #                 newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
            #                 dataframe_list.append(newobj)
            func_dict[func_name] = dataframe_list
        #         for key, val in func_dict:
        #             newobj = DataFrame()
        #             newobj.func_name = key
        #             newobj.exclusive_runtime = self.exclusive_runtimes_list[hashindex]
        #             newobj.severity = self.severity_list[hashindex]
        #             newobj.callstack_hash = self.hashed_call_stack_list[hashindex]
        #             dataframe_list.append(newobj)
        #             prev = hashindex +1 ##So that, the next position starts from hashindex+1
        #         return set(unique_func_names), set(unique_callpath_hash), dataframe_list
        return func_dict

    def get_unique_callpaths_from_a_list(self, dataframe_list):
        names = []
        for obj in dataframe_list:
            names.append(obj.callstack_hash)
        #         print("Unique call stacks in a list: ", set(names))
        return set(names)

    def get_number_of_callpath_with_norm_severity(
        self, exclusive_lower_limit, inclusive_upper_limit
    ):
        res = [
            idx
            for idx, val in enumerate(self.norm_severity_list)
            if (val > exclusive_lower_limit) and (val <= inclusive_upper_limit)
        ]
        unique_callpath_hash = [self.hashed_call_stack_list[i] for i in res]
        num_unique_elem = len(set(unique_callpath_hash))
        return num_unique_elem

    def get_number_of_funcs_with_norm_severity(
        self, exclusive_lower_limit, inclusive_upper_limit
    ):
        res = [
            idx
            for idx, val in enumerate(self.norm_severity_list)
            if (val > exclusive_lower_limit) and (val <= inclusive_upper_limit)
        ]
        unique_func_names = [self.func_names_list[i] for i in res]
        num_unique_elem = len(set(unique_func_names))
        return num_unique_elem

    def get_unique_ranks_list(self):
        unique_ranks_list = set(self.rank_list)
        return list(unique_ranks_list)

    def get_counts_norm_severity(self):
        return Counter(self.norm_severity_list)

    def get_counts_severity(self):
        return Counter(self.severity_list)

    def get_counts_hash_values(self):
        return Counter(self.hashed_call_stack_list)

    def plot_histogram_callstacks(self):
        fig = go.Figure(data=[go.Histogram(x=self.hashed_call_stack_list)])
        fig.show()

    def plot_histogram_ranks(self):
        fig = go.Figure(data=[go.Histogram(x=self.rank_list)])
        fig.show()

    def get_callstack_map(self, callstack_hash):
        if callstack_hash not in self.callstack_map:
            self.callstack_map[callstack_hash] = (
                len(self.callstack_map) + 1
            )  # new index = number of current elem + 1
        return self.callstack_map[callstack_hash]  # whatever was the index

    def get_callgraph_edgelist_for_function(self, func_name):
        return None

    def create_callgraph_edgelist_for_function(
        self, func_name, callgraph_list_of_pairs
    ):
        if func_name not in self.callgraph_edgelist_per_function:
            self.callgraph_edgelist_per_function[func_name] = []
        # for key, val in callgraph_dict.items():
        for v1, v2 in callgraph_list_of_pairs:
            print(v1, v2)
            ############### Add code to add edges

        return None

    def plot_histogram_severity(self):
        fig = go.Figure(data=[go.Histogram(x=self.norm_severity_list)])
        fig.show()

    def plot_sunburst_chart(self, severity_bins=[0.9]):  # (funcs, parents, values):
        ids = []
        labels = []
        parents = []
        values = []
        hover_labels = []
        pairs = []

        PERCENT_OFFSET = 1.00001
        for sev in severity_bins:
            runtimes = OrderedDict()  # {}
            callstack_runtimes = OrderedDict()

            func_dict = self.return_perfunc_entries_for_specific_norm_severity(
                sev - 0.1, sev
            )
            number_of_callpaths = self.get_number_of_callpath_with_norm_severity(
                sev - 0.1, sev
            )

            total_execution_time_across_funcs = 0
            func_name_list = []
            callstack_dict = OrderedDict()
            for func_name, dataframe_list in func_dict.items():
                total_time_in_func = 0
                func_name_list.append(func_name)

                for obj in dataframe_list:  # for each call path, may not be unique
                    if obj.callstack_hash in callstack_runtimes.keys():
                        callstack_runtimes[obj.callstack_hash] += obj.exclusive_runtime
                    else:
                        callstack_runtimes[obj.callstack_hash] = obj.exclusive_runtime

                    total_time_in_func += (
                        sev * obj.exclusive_runtime
                    )  # for each call path
                    total_execution_time_across_funcs += sev * obj.exclusive_runtime
                runtimes[func_name] = total_time_in_func
                unique_callstack_hash_set = self.get_unique_callpaths_from_a_list(
                    dataframe_list
                )
                callstack_dict[func_name] = list(unique_callstack_hash_set)

            ## This block is for normalizing the function runtimes
            normed_runtime = OrderedDict()  # {}
            # Assures the sum of normed_runtime is less than 1
            runtime_sum = sum(runtimes.values()) * PERCENT_OFFSET
            for func_name in func_name_list:
                normed_runtime[func_name] = runtimes[func_name] / runtime_sum

            # This block normalizes each callstack's average runtime for each function
            normed_callstack_runtime = OrderedDict()  # {}
            # Assures the sum of normed_runtime is less than 1
            runtime_sum = sum(callstack_runtimes.values()) * PERCENT_OFFSET
            for func_name in func_name_list:
                normed_callstack_runtime[func_name] = {}
                for callstack in callstack_dict[func_name]:
                    normed_callstack_runtime[func_name][callstack] = (
                        callstack_runtimes[callstack] / runtime_sum
                    )

            app_name = str(sev)
            # First layer
            # Add the app name and its value is the sum of the normed runtime
            ids.append(app_name)
            pairs.append([app_name])
            labels.append(app_name)
            hover_labels.append(app_name)
            parents.append("")
            values.append(
                sum(normed_runtime.values())
            )  # How do we take into account the # of func count?

            # Second layer
            # Consists of each region whose value is their runtime percentage
            hover_label = "%s<br>Runtime: %0.2f%%"
            for reg in func_name_list:
                ids.append(reg + app_name)
                pairs.append([reg])
                labels.append(reg)
                hover_labels.append(hover_label % (reg, normed_runtime[reg] * 100.0))
                parents.append(app_name)
                values.append(normed_runtime[reg])

            # Third Layer
            # Consists of each resource per a particular region
            #             for k, v in normed_callstack_runtime.items():
            #                 for kk, vv in v.items():
            #                     print(k, kk, vv)

            # First step is to make a belief mapping
            belief_map = OrderedDict()  # {}
            for reg in func_name_list:
                belief_map[reg] = {}
                # for resource, res_percent_err in rsm_results[reg].items():
                for resource in callstack_dict[reg]:  # For each unique callstack
                    belief_map[reg][resource] = normed_callstack_runtime[reg][
                        resource
                    ]  # res_percent_err

            # Next step is with these belief values, normalize them such that
            # the sum of all beliefs is equal to the region's normalized value
            # We do this by first dividing all values by the sum
            # This assures that the sum of these values equals 1
            normed_belief_map = {}
            for reg in func_name_list:
                normed_belief_map[reg] = {}
                belief_sum = sum(belief_map[reg].values()) * PERCENT_OFFSET
                for resource, belief in belief_map[reg].items():
                    normed_belief_map[reg][resource] = belief / belief_sum

            # Lastly we append everything to the sunburst
            hover_label = "%s<br>Percent Contribution: %0.2f%%"
            for reg in func_name_list:
                for resource, belief in belief_map[reg].items():
                    ids.append(reg + resource + app_name)
                    pairs.append([reg, resource])
                    labels.append("CS" + str(self.get_callstack_map(resource)))
                    hover_labels.append(
                        hover_label
                        % (resource, normed_belief_map[reg][resource] * 100.0)
                    )
                    parents.append(reg + app_name)
                    values.append(
                        normed_belief_map[reg][resource] * normed_runtime[reg]
                    )
        sunburst_colors = ["#FFFFFF"]
        default_color = "#babbca"  #'#636efa'

        ################################################################################

        trace = go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertext=hover_labels,
            hoverinfo="text",
            insidetextfont={"size": 20, "color": "#000000"},
            outsidetextfont={"size": 30, "color": "#377eb8"},
            #        outsidetextfont = {"size": 20, "color": "#377eb8"},
            marker={"line": {"width": 2}},
            # marker_colors=sunburst_colors
        )
        ############################################################################
        layout = go.Layout(
            margin=go.layout.Margin(t=0, l=0, r=0, b=0),
        )

        #             templates = [
        #                 "bootstrap",
        #                 "minty",
        #                 "pulse",
        #                 "flatly",
        #                 "quartz",
        #                 "cyborg",
        #                 "darkly",
        #                 "vapor",
        #             ]

        template = "minty"  # "minty"#"ggplot2"
        load_figure_template(template)
        fig = go.Figure([trace], layout)
        fig.update_layout(width=400, height=400, template=template)

        return fig

    def get_short_hash_value(self, hashvalue, N=8):
        if isinstance(hashvalue, int):
            return hashvalue % 1000000
        #         print("Last ", N, " digits of hashvalue", hashvalue[-N:])
        return hashvalue[-N:]

    def get_hash(self, strings):
        """
        This function ensures that the hash of ['string1', 'string2'] will be different from the has of ['string1str', 'ing2'],
        as well as ['string1string2'].
        """
        tohash = repr(strings)
        result = hashlib.sha256(tohash.encode("utf-8")).hexdigest()
        return result


"""## TREE DEF"""
## Call this function to create a nested dictionary
import collections


def tree():
    return collections.defaultdict(tree)


"""## GET ALGO PARAMETERS"""


def get_algo_params(d):
    out_dict = {}
    for key, val in d.items():
        if isinstance(val, dict):
            for k, v in val.items():
                #print("--", k, v)
                out_dict[k] = v
        else:
            #print(key, val)
            out_dict[key] = val
    return out_dict


def get_call_stack_dict(l):
    out_dict = {v["entry"]: v for v in l}
    root = 0
    prev_func = ""
    call_stack = []
    for key, val in out_dict.items():
        if root == 0:  # if root of the callgraph
            prev_func = val["func"]
        else:
            call_stack.append((val["func"], prev_func))
            prev_func = val["func"]
        root += 1
    return call_stack


def get_call_stack_dict_old(l):
    out_dict = {v["entry"]: v for v in l}

    call_stack = [(v["func"] + "->" + str(v["is_anomaly"])) for v in l]
    return call_stack


"""## NORMALIZED DATA"""


def normalize_1d(data):
    min_val = min(data)
    max_val = max(data)
    return [(d - min_val) / (max_val - min_val) for d in data]


def is_func_in_group(func_name, function_groups_list):
    parent = ""
    for grp in function_groups_list:
        if grp in func_name:
            parent = grp
            break
    return parent


"""## CONVERT JSON TO DATAFRAME """


########### These two will go in the driver.py file.
def convert_json_to_dataframe(dict_entries):
    obj = DataFrame()
    for entry_idx in np.arange(len(dict_entries)):
        func_name = dict_entries[entry_idx]["func"]
        obj.func_names_list.append(func_name)
        obj.hashed_call_stack_list.append(
            str(
                obj.get_short_hash_value(
                    obj.get_hash(dict_entries[entry_idx]["call_stack"]), 6
                )
            )
        )
        obj.exclusive_runtimes_list.append(dict_entries[entry_idx]["runtime_exclusive"])
        obj.severity_list.append(dict_entries[entry_idx]["outlier_severity"])
        obj.rank_list.append(dict_entries[entry_idx]["rid"])

        # Callgraph edge list making
        obj.create_callgraph_edgelist_for_function(
            func_name, dict_entries[entry_idx]["call_stack"]
        )
    #         obj.entry_time = dati.fromtimestamp(dict_entries[entry_idx]["entry"])
    #         obj.exit_time = dati.fromtimestamp(dict_entries[entry_idx]["exit"])
    obj.unique_function_names = list(set(obj.func_names_list))
    obj.norm_severity_list = normalize_1d(obj.severity_list)
    obj.norm_exclusive_runtimes_list = normalize_1d(obj.exclusive_runtimes_list)
    return obj

def local_df_to_pandas(obj):
    df = pd.DataFrame(list(zip(obj.func_names_list, obj.hashed_call_stack_list, obj.norm_severity_list)),
                      columns=["func_names_list", "hashed_call_stack_list", "norm_severity_list"])
    df["hashed_callstack - func_name"] = df["hashed_call_stack_list"].map(str) + "-" + df["func_names_list"]
    df.drop(['func_names_list', 'hashed_call_stack_list'], axis = 1, inplace = True)
    return df


"""## TRAVERSE"""
####### REference: https://nvie.com/posts/modifying-deeply-nested-structures/


def traverse(obj, path=None):
    if path is None:
        path = []

    if isinstance(obj, dict):
        return {k: traverse(v, path + [k]) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [traverse(elem, path + [[]]) for elem in obj]
    elif isinstance(obj, tuple):
        return [traverse(elem, path + [[]]) for elem in obj]
    elif isinstance(obj, int):
        return int(obj)
        # return np.int(obj)
    elif isinstance(obj, float):
        return float(obj)
        # return np.float(obj)

    else:
        return obj


"""## READ JSON FILE"""


def read_json_file(filename):
    dict_entries = tree()
    # Opening JSON file
    # with open('dump_shard0.json') as json_file:
    with open(filename) as json_file:
        dict_data = json.load(json_file)
        for entry_idx in np.arange(len(dict_data)):
            dd = traverse(dict_data[entry_idx])  # gets converted to a dict
            func_name = ""
            for key in dd.keys():
                if key == "call_stack":
                    dict_entries[entry_idx][key] = get_call_stack_dict(dd[key])
                else:
                    dict_entries[entry_idx][key] = traverse(dd[key])
                if key == "func":
                    func_name = dict_entries[entry_idx][key]
    return dict_entries


def find_bound(x):
    bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bounds) - 1):
        if bounds[i] <= x < bounds[i + 1]:
            return bounds[i + 1]
        if x == bounds[-1]:
            return bounds[-1]
    return None


def truncate_decimals(df, column_name, new_column_name):
    df[new_column_name] = df[column_name].apply(lambda x: int(x * 100) / 100)
    return df


def apply_upper_bound(df, column_name, bounds):
    new_column_name = "sev_bin_upper_bound"

    df[new_column_name] = df[column_name].apply(find_bound)
    return df



def basic_info_to_df(obj, output_file):

    #BUILD NEW DATAFRAME WITH ALL REQUIRED INFORMATION: NUMBER OF CALLS OF EACH FUNCTION, FUNCTION ID/CALL STACK, SEVERITY
    data_dict={'all_func_names': [], 'num_calls': [], 'call_stack':[], 'severity':[], 'exclusive_runtimes':[]}
    for func_name in obj.func_names_list:
        data_dict['all_func_names'].append(func_name)
    for hashed_call_stack in obj.hashed_call_stack_list:
        data_dict['num_calls'].append(obj.hashed_call_stack_list.count(hashed_call_stack))
        data_dict['call_stack'].append(hashed_call_stack)

    for exclusive_runtime in obj.exclusive_runtimes_list:
        data_dict['exclusive_runtimes'].append(exclusive_runtime)

    for normalized_sev in obj.norm_severity_list:
        data_dict['severity'].append(normalized_sev)

    for entry_time in obj.entry_time_list:
        data_dict['entry_time'].append(entry_time)
        print(entry_time)



    df = pd.DataFrame(data_dict)
    df = df.drop_duplicates(subset=['call_stack', 'severity'])
    total_runtime=[sum(obj.exclusive_runtimes_list)]



    #append empty spaces so it can output to csv properly
    for i in range(len(df)-1):
        total_runtime.append('')
    df['TotalRuntimeSum(AllFunc)']=total_runtime
    df.set_index('call_stack', inplace=True)

    severity_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df = truncate_decimals(df, 'severity', 'severity_trunc')
    df = apply_upper_bound(df, 'severity_trunc', severity_bins)
    # Useful values(yes, it breaks without the 0.0)
    df.to_csv(output_file)
def extended_info_to_csv(file_name_limit):
    possible_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #if n=10
    #if s = exclusive_runtimes
    for file_name in os.listdir("unfiltered_inp_files/"):
        df = pd.read_csv("unfiltered_inp_files/" + file_name)
        #df.drop_duplicates(subset='call_stack', inplace=True)
        df.sort_values(by = "exclusive_runtimes", ascending = False, inplace=True)
        df.drop_duplicates(subset=['call_stack', 'num_calls'], inplace=True)
        df.reset_index(inplace=True)
        index=0
        for i in range(len(df["exclusive_runtimes"].tolist())):
            if df["exclusive_runtimes"].tolist()[index]==0:
                df.drop(labels=i, inplace=True)
                index-=1
            index+=1
        df.to_csv("filtered_inp_files/only_rem_duplicates/" + file_name[0:file_name_limit] + ".csv")
        df=df.head(10)

        #print(df)
        df.reset_index(inplace=True)
        df.to_csv("filtered_inp_files/unshortened/" + file_name[0:file_name_limit] + ".csv")
    df = pd.concat(map(pd.read_csv, ["filtered_inp_files/unshortened/"+ i for i in os.listdir("filtered_inp_files/unshortened/")]), ignore_index=True)
    df=df.drop(['Unnamed: 0', 'index', 'all_func_names', 'severity_trunc', 'TotalRuntimeSum(AllFunc)', 'num_calls', 'severity', 'sev_bin_upper_bound'], axis='columns')
    df.drop_duplicates(subset='call_stack', inplace=True)
    df.sort_values(by="exclusive_runtimes", ascending=False, inplace=True)

    #can be removed, but it's just to lessen the load:

    df = df.head(10)

    df.to_csv("filtered_inp_files/func_data/combined_func_data.csv")
    print(df)



    """sev_bins_column = []
    for i in possible_bins:
        sev_bins_column.append(i)
    for i in range(len(df) - len(possible_bins)):
        sev_bins_column.append(' ')
    agg_runtime_column = []
    for bin in possible_bins:
        aggregated_runtime = 0
        for i in range(len(df["sev_bin_upper_bound"])):
            if (df["sev_bin_upper_bound"][i] == bin):
                aggregated_runtime += df["exclusive_runtimes"][i]
        agg_runtime_column.append(aggregated_runtime)
    for i in range(len(df) - len(possible_bins)):
        agg_runtime_column.append(" ")
    df["all_sev_bins"] = sev_bins_column
    df["agg_runtimes"] = agg_runtime_column
    if add_identifier==True:

        df.to_csv("filtered_inp_files/unshortened/" + file_name[0:2] + " "+ str(index)+".csv")
    else:
        df.to_csv("filtered_inp_files/unshortened/" + file_name[0:2] + ".csv")
    index+=1"""

def containsLetterAndNumber(input):
    return input.isalnum() and not input.isalpha() and not input.isdigit()

def data_scaler_median(data):
    output=[]
    median=statistics.median(data)
    minimum = min(data)
    maximum = max(data)

    for data_point in data:
        if data_point<=median:
            output_val=(data_point-minimum)/(median-minimum)
            output_val*=0.5
            output.append(output_val)

        else:
            output_val = (data_point-median)/(maximum-median)
            output_val*=0.5
            output_val+=0.5
            output.append(output_val)
    return output

def data_scaler_mean(data):
    output=[]
    mean=statistics.mean(data)
    minimum = min(data)
    maximum = max(data)

    for data_point in data:
        if data_point<=mean:
            output_val=(data_point-minimum)/(mean-minimum)
            output_val*=0.5
            output.append(output_val)

        else:
            output_val = (data_point-mean)/(maximum-mean)
            output_val*=0.5
            output_val+=0.5
            output.append(output_val)
    return output


def min_max_scaler(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled

def get_chord_info():
    possible_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for file_name in os.listdir("filtered_inp_files/unshortened/"):
        print("Reading file: " + str(file_name))
        df = pd.read_csv("filtered_inp_files/unshortened/" + file_name)

        new_df = pd.DataFrame()
        unique_call_stacks = []
        severity = []
        total_runtimes = []
        # unique_call_stacks=
        df = df.drop_duplicates(subset=['call_stack', 'sev_bin_upper_bound'])
        df.reset_index(inplace=True)

        unique_call_stacks = df['call_stack']

        unique_call_stacks = unique_call_stacks.tolist()

        df.reset_index(inplace=True)
        index = 0
        for unique_call_stack in unique_call_stacks:

            total_runtime = 0
            for i in range(len(df['call_stack'])):
                if unique_call_stack == df['call_stack'][i]:
                    total_runtime += df.iloc[i].get(key='exclusive_runtimes')
            total_runtimes.append(total_runtime)
            df.set_index('call_stack', inplace=True)

            severity.append(df['sev_bin_upper_bound'][index])

            df.reset_index(inplace=True)
            index += 1

        new_df['unique_call_stacks'] = unique_call_stacks
        new_df['total_runtimes'] = total_runtimes
        # new_df['normalized_runtimes_with_median']=data_scaler_median(total_runtimes)
        # new_df['normalized_runtimes_with_mean'] = data_scaler_mean(total_runtimes)

        # new_df['normalized_runtimes_with_minmax'] = min_max_scaler(new_df['total_runtimes'].values.reshape(-1, 1))
        new_df['severity'] = severity
        total_runtime_sum_list = []
        for i in range(len(unique_call_stacks)):
            total_runtime_sum_list.append('')
        total_runtime_sum_list[0] = df.iloc[0]["TotalRuntimeSum(AllFunc)"]
        new_df['TotalRuntimeSum(AllFunc)'] = total_runtime_sum_list
        new_df.to_csv("filtered_inp_files/chord_data/" + str(file_name)[0:2] + "_CHORD_DATA.csv")


def main():

    """

    Command Line Params:

    python setup.py input_folder add_identifier

    input_folder (required): represents folder containing input JSON files

    add_identifier (optional): a number (1 or 0) that is used to distinguish between output files.
                    1 means that program will add an identifier (index) to the end of each file name, and 0 means that program will not.
                    Use this option if you are comparing the same file to one another, so the program will output 2 different files instead of replacing the old one



    """




    """input_file_list = [
        "inp_for_circos/short-Grid-Xconj_evol_16c-repeat_1rank-chimbuko_304740-provdb.json",
        "inp_for_circos/short-Grid-Xconj_evol_16c-repeat_1rank-chimbuko_306239-provdb.json",
        "inp_for_circos/short-Grid-Xconj_evol_16c-repeat_1rank-chimbuko_309131-provdb.json",
    ]"""
    input_file_list = [
        "inp_for_circos/Grid-Xconj_evol_16c-repeat_1rank-chimbuko_304740-provdb.json",
        "inp_for_circos/Grid-Xconj_evol_16c-repeat_1rank-chimbuko_306239-provdb.json",
        "inp_for_circos/Grid-Xconj_evol_16c-repeat_1rank-chimbuko_309131-provdb.json",
    ]

    """input_file_list = [
        "inp_for_circos/Grid-Xconj_evol_16c-model_reuse-chimbuko_306104-provdb.json",
        "inp_for_circos/Grid-Xconj_evol_16c-model_reuse-chimbuko_306124-provdb.json",
        "inp_for_circos/Grid-Xconj_evol_16c-model_reuse-chimbuko_306253-provdb.json",
        "inp_for_circos/Grid-Xconj_evol_16c-model_reuse-chimbuko_309126-provdb.json"
    ]"""
    """input_file_list = [
        "inp_for_circos/Grid-Xconj_evol_16c-repeat_1rank-chimbuko_304740-provdb.json",
        "inp_for_circos/Grid-Xconj_evol_16c-repeat_1rank-chimbuko_304740-provdb.json"
    ]"""
    for i in range(len(input_file_list)):
        print("READING JSON FILE: " + str(input_file_list[i])+"\n")
        dict_entries = read_json_file(input_file_list[i])

        print("CONVERTING TO DATAFRAME\n")
        obj = convert_json_to_dataframe(dict_entries)
        #basic_info_to_df(obj, 'filtered_inp_files/'+input_file_list[i][15:75]+'.csv')

        print("OUTPUTTING FILES TO UNFILTERED INP FILES FOLDER\n")
        basic_info_to_df(obj, 'unfiltered_inp_files/' + input_file_list[i][61:63] +'.csv')
        #basic_info_to_df(obj, 'unfiltered_inp_files/' + input_file_list[i][59:62] + '.csv')
    print("CALCULATING EXTENDED INFORMATION\n")
    extended_info_to_csv(2)

    #print("GETTING CHORD DATA\n")
    #get_chord_info()
main()
