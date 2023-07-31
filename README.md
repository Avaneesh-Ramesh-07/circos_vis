#Circos
Circos is a visualization that maps relationships between data organized in a circular layout. For this framework, we used the Circos module from the Dash Plotly Library

#Running
This framework is designed to take in multiple dumps of JSON data and graph the relations between them. To produce visualization, one must first run the setup.py file, which configures and adds all the necessary files to generate the visualization

This script takes two parameters: input_folder and add_identifier.

The required parameter, input_folder, is the folder that contains the JSON files that are parsed and used to produce visualization.

The optional parameter, add_identifier, is a number (0 or 1) that  is used to determine whether an index identifier should be placed at the end of each file name. This is particularly useful when visualizing the same file against itself. 

To run the setup script, simply run the command:

```mermaid
python setup.py input_folder add_identifier
```

and provide the necessary parameters.



The second step is to run the vis.py file which produces the visualization using the files produced from the setup.py.

This script takes in 2 optional parameters: top_num_func and file_delim.

The parameter, top_num_func, is a number that represents the top number of functions to display from each severity bin. Default value for this parameter is 5.

The parameter, file_delim, is a number that represents a file delimeter to accurately interpret files (will recognize '30' in filename 30_func_and_runtime.csv). Default value for this parameter is 2.

To run the vis script, simply run the command:

```mermaid
python vis.py top_num_func file_delim
```

and provide the necessary parameters.

