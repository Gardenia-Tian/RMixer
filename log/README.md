# Log Folder

This folder contains logs and related scripts for managing and analyzing logs generated by the client and server components.

## Contents

- **client.log:** Log file for client-related information.
- **server.log:** Log file for server-related information.
- **draw_log.py:** Script for visualizing the running process based on logs.
- **draw_log.sh:** Batch script for running `draw_log.py`.
- **get_device_workload.py:** Script to obtain the load of each device based on logs.
- **get_time.py:** Script to obtain the average running time of tasks.
- **postprocess.py:** Postprocessing script for the logs.
- **process_all_log.sh:** Batch script for processing all logs.

## Usage

To use the log-related scripts, follow these steps:

1. **Visualize Running Process:**
   
   - Run the `draw_log.sh` script to visualize the running process based on logs.
   
   ```bash
   bash draw_log.sh filename target_folder
   
2. **Obtain Device Workload:**

   - Execute the `get_device_workload.py` script to obtain the load of each device based on logs.

   ```bash
   python get_device_workload.py input_folder output_folder
   ```
   
3. **Obtain Average Running Time:**

   - Run the `get_time.py` script to obtain the average running time of tasks.

   ```bash
   python get_time.py input_folder output_folder
   ```
   
4. **Postprocess Logs:**

   - Execute the `postprocess.py` script for postprocessing the logs.

   ```shell
   bashCopy code
   python postprocess.py output_name
   ```

5. **Batch Process All Logs:**

   - Run the `process_all_log.sh` script to process all logs in batches.

   ```bash
   ./process_all_log.sh input_folder output_folder
   ```
