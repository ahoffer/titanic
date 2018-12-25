# Key: A key is an entry in the DKV that maps to an object in H2O.
# Frame: A Frame is a collection of Vec objects. It is a 2D array of elements.
# Vec: A Vec is a collection of Chunk objects. It is a 1D array of elements.
# Chunk: A Chunk holds a fraction of the BigData. It is a 1D array of elements.
# ModelMetrics: A collection of metrics for a given category of model.
# Model: A model is an immutable object having predict and metrics methods.
# Job: A Job is a non-blocking task that performs a finite amount of work.

import h2o

h2o.init()

