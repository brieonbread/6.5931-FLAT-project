# Project #2: Investigating the Trade-off Between Sequential and Pipelined Fusion
When fusing layers, processing elements (PEs) can be assigned to all processes one layer at a
time sequentially or in a pipeline. In this project, you compare sequential and pipelined versions
of an accelerator using the same dataflow. Finally, you answer how architecture choices differ
when optimized for sequential or pipelined fusion.

## Quick Start
Choose one of `docker-compose-looptree.yaml` or `docker-compose-timeloop.yaml`  and rename to `docker-compose.yaml`. Make sure to update the file as needed.

Then, run the following command.

```
docker compose up
```

## Resources
- [The LoopTree paper](https://arxiv.org/abs/2409.13625).
- [Timeloop/Accelergy v4 documentation](https://timeloop.csail.mit.edu/v4).
- [The PyTimeloop code](https://github.com/Accelergy-Project/timeloop-python).
- [The Timeloop code](https://github.com/NVlabs/timeloop).

