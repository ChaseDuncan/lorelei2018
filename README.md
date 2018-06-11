### LORELEI 2018 EDL

Currently the project is broken down into three directories: *utils*, *models*, and *scripts*. The *utils* directory is for utilities which can be shared across multiple modules or models. Currently it is primarily made up of two files, io_utils.py and data_utils.py. The former broadly is for utilities which write and read from disk while the latter is for operating on data.

preprocess.py is for any preprocessing of the data that needs to be done. evalution.py is basically deprecated as we move to use the LORELEI evaluation metrics but it is used for calculating various accuracy scores. The rest of the files in *utils* are WIP stuff for coherence which can be safely ignored by anyone not directly involved in that project.

The *models* directory is where an model that is used in the project should go

*scripts* are for everything else that doesn't fit into the above.
