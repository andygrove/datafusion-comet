#!/bin/bash

export SPARK_MASTER=spark://localhost:7077

$SPARK_HOME/bin/spark-submit \
    --master $SPARK_MASTER \
    --conf spark.driver.memory=8G \
    --conf spark.executor.instances=1 \
    --conf spark.executor.cores=8 \
    --conf spark.cores.max=8 \
    --conf spark.executor.memory=8g \
    --conf spark.memory.offHeap.enabled=true \
    --conf spark.memory.offHeap.size=8g \
    --conf spark.local.dir=/home/ec2-user/tmp \
    --conf spark.driver.extraJavaOptions="-Djava.io.tmpdir=/home/ec2-user/tmp" \
    --conf spark.executor.extraJavaOptions="-Djava.io.tmpdir=/home/ec2-user/tmp" \
    tpcbench.py \
    --name spark \
    --benchmark tpch \
    --data /home/ec2-user/ \
    --queries /home/ec2-user/datafusion-benchmarks/tpch/queries \
    --output . \
    --iterations 1
