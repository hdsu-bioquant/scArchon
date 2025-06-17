#!/bin/bash

# Define the output log file
log_file=$1

# Start time tracking
start_time=$(date +%s)

# Run the command passed as arguments
"$@"

# End time tracking
end_time=$(date +%s)
runtime=$((end_time - start_time))

# Log the runtime to the file
echo "Runtime (seconds): $runtime" > "$log_file"

# Monitor GPU memory usage and log it
if command -v nvidia-smi &> /dev/null; then
    max_gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -n1)
    echo "Max GPU Memory Used (MiB): $max_gpu_memory" >> "$log_file"
else
    echo "GPU monitoring skipped: nvidia-smi not found." >> "$log_file"
fi

# Monitor CPU memory usage in GB
max_cpu_memory_kb=$(grep MemTotal /proc/meminfo | awk '{{print $2}}')
used_cpu_memory_mb=$(free -m | awk '/^Mem:/ {{print $3}}')

# Convert CPU memory from kB to GB and from MiB to GB
max_cpu_memory_gb=$((max_cpu_memory_kb / 1048576))
used_cpu_memory_gb=$((used_cpu_memory_mb / 1024))

echo "Total CPU Memory (GB): $max_cpu_memory_gb" >> "$log_file"
echo "Used CPU Memory (GB): $used_cpu_memory_gb" >> "$log_file"
