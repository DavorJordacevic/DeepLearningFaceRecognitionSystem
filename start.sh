#!/bin/bash

rm nohup.out
nohup python main.py --cdp=config.json 2>&1 &
echo $! > save_pid.pid

echo "FRAPP starter successfully with PID: $!"