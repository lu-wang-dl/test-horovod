# Databricks notebook source
import horovod.tensorflow.keras as hvd

def run_training_horovod():
    # Horovod: initialize Horovod.
    hvd.init()
    import os
    print(os.environ.get('PYTHONPATH'))
    print(os.environ.get('PYTHONHOME'))
    print(f"Rank is: {hvd.rank()}")
    print(f"Size is: {hvd.size()}")

# COMMAND ----------

from sparkdl import HorovodRunner

hr = HorovodRunner(np=-spark.sparkContext.defaultParallelism, driver_log_verbosity="all")
hr.run(run_training_horovod)

# COMMAND ----------

from sparkdl import HorovodRunner

hr = HorovodRunner(np=spark.sparkContext.defaultParallelism, driver_log_verbosity="all")
hr.run(run_training_horovod) # manually stopping b/c it's just hanging

# COMMAND ----------


