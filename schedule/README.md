# Schedule Folder

This folder contains the scheduling components of the project. It is responsible for managing job scheduling, device allocation, and related functionalities.

## Contents

- **device_context.py:** Module for managing device contexts.
- **device_policy.py:** Module defining device allocation policies.
- **frontend_schedule.py:** Module for scheduling system.
- **frontend_tcp.py:** TCP communication module for the scheduling frontend.
- **job_context.py:** Module for managing job contexts.
- **job_policy.py:** Module defining job scheduling policies.
- **main.py:** Main entry point for the scheduling system.
- **run_server.sh:** Shell script for running the scheduling server.
- **worker_common.py:** Common functionality shared by scheduling workers.
- **worker.py:** Scheduling worker module.

## Usage

To use the scheduling components, follow these steps:

1. **Customize Policies Configuration: **

   - In the `run_server.sh` file in the `rmixer` directory, you can configure the task list, number of workers, scheduling policy, and other parameters. You can select the configuration according to your requirements.

2. **Run the Server:**

   - Execute the `run_server.sh` script to start the scheduling server.

   ```bash
   bash ./rmixer/run_server.sh

3. **Interact with the Frontend:**

   - Use the `run_client.sh` module to interact with the scheduling system.

   ```bash
   bash ./client/run_client.py
   ```




