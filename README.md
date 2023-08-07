#Circos
Circos is a visualization that maps relationships between data organized in a circular layout. For this framework, we used the Circos module from the Dash Plotly Library

#Running
This framework is designed to take in multiple dumps of JSON data and graph the relations between them. To produce visualization, one must first run the setup.py file, which configures and adds all the necessary files to generate the visualization

To run the setup script, simply run the command:

```mermaid
python setup.py file_name_delimiter
```
This takes one parameter explained below:

- input folder (required): path to input folder of json dump files


The second step is to run the vis.py file which produces the visualization using the files produced from the setup.py.

To run the vis script, simply run the command:

```mermaid
python vis.py force_compute file_name_limit
```

This script takes in 1 parameter: force_compute (-f):

- force_compute: optional flag (-f) that tells the program whether to force compute the information for visualization or whether to use cached data. If this is not included, the program will use cached data.