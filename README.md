#Circos
Circos is a visualization that maps relationships between data organized in a circular layout. For this framework, we used the Circos module from the Dash Plotly Library

#Running
This framework is designed to take in multiple dumps of JSON data and graph the relations between them. To produce visualization, one must first run the setup.py file, which configures and adds all the necessary files to generate the visualization

To run the setup script, simply run the command:

```mermaid
python setup.py 
```




The second step is to run the vis.py file which produces the visualization using the files produced from the setup.py.

This script takes in 2 parameters: force_compute (-f) and file_name_limit.

The parameter, force_compute, is an optional flag (-f) that tells the program whether to force compute the information for visualization or whether to use cached data. If this is not included, the program will use cached data.

The parameter, file_name_limit, is a number that represents a file delimeter to accurately interpret files (will recognize '30' in filename 30_func_and_runtime.csv). Default value for this parameter is 2.

To run the vis script, simply run the command:

```mermaid
python vis.py force_compute file_name_limit
```

and provide the necessary parameters.

