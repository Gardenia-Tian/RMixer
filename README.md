# RMixer: Sophisticated Orchestrating Concurrent DLRM Training on CPU/GPU Platform

![RMixer framwork](Figures/framwork.png)

## Overview

RMixer is a scheduling framework designed to optimize training of Deep Learning Recommendation Models (DLRM) on CPU-GPU platforms. This framework aims to enhance resource utilization and throughput, providing a valuable tool for both research and practical implementations.

## Features

- **Sophisticated Scheduling:** RMixer employs advanced scheduling techniques to orchestrate concurrent DLRM training on CPU and GPU.

- **Improved Resource Utilization:** The framework enhances resource efficiency, maximizing the utilization of both CPU and GPU resources.

- **High Throughput:** RMixer is designed to boost the training throughput for recommendation models, resulting in faster model convergence.

## Quick Start

### Conda environment

If using multiple GPU, make sure that NCCL>=2.14.3 is installed in your device.

To get started with RMixer, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Gardenia-Tian/RMixer.git
   cd RMixer
   ```

2. **Create and activate environment:**

   ```shell
   conda env create -f environment.yml
   conda activate rmixer
   ```

3. **Download datasets:**

   ```bash
   cd datasets
   bash download.sh
   ```

4. **Usage:**

   ```bash
   cd ../comp/rmixer
   bash run.sh
   ```

### Docker

```bash
# Search for rmixer image
docker search rmixer
# Pull the image
docker pull rmixer:v0.4
# Check whether the image is successfully pulled
docker images
# Start the container
docker run --name rmixer --gpus all -it rmixer:v0.4 /bin/bash
```

## Directory structure

```
├── comp                               # The run script and the run result
│   ├── monolithic                         # The runtime script of monolithic 
│   ├── mps                                # The runtime script of mps
│   └── rmixer                             # The runtime script of rmixer 
├── datasets                           # The data sets used are specified in this directory
│   ......
│   └── README.md
├── environment.yml                    # Environment configuration
├── Figures                            # Figures related to the project
│   ├── framwork.png
│   └── README.md
├── LICENSE                           
├── log                                # Log related directory
│   ├── client.log                         # Client log
│   ├── server.log                         # Server log
│   ├── draw_log.py                        # Visualize the running process based on logs
│   ├── draw_log.sh                        # Batch script for draw_log.py
│   ├── get_device_workload.py             # Obtain the load of each device based on logs
│   ├── get_time.py                        # Obtain the average running time of tasks
│   ├── postprocess.py                     # postprocess for the log
│   ├── process_all_log.sh                 # Process all logs in batches
│   └── README.md
├── models                            # DLRM for evaluation
│   ......
│   └── README.md
├── README.md                         # Documentation for RMixer
└── schedule                          # The main code of RMixer
    ├── client                            # Contains client code
    ├── data                              # Task list
    ├── rmixer                            # RMixer server code
    ├── util                              # Some other tools
    └── README.md
```



## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for research and practical purposes.

## Acknowledgments

- Special thanks to Dr. Huang for his valuable guidance and mentorship throughout the project.

- We would like to express our sincere gratitude to the authors of "DeepRecSys: A System for Optimizing End-To-End At-scale Neural Recommendation Inference" for their groundbreaking work and inspiration. The DeepRecSys repository can be found at [https://github.com/harvard-acc/DeepRecSys](https://github.com/harvard-acc/DeepRecSys).

- Special thanks to the authors of "CoGNN: efficient scheduling for concurrent GNN training on GPUs" for their insightful research and ideas that have greatly influenced our project. The CoGNN repository is available at [https://github.com/guessmewho233/CoGNN_info_for_SC22](https://github.com/guessmewho233/CoGNN_info_for_SC22).

- We extend our gratitude to the open-source community for their contributions and feedback.

## Contact

For any inquiries, issues, or collaboration opportunities, please contact Rui at tianr6@mail2.sysu.edu.cn.
