#!/bin/bash
nvidia-smi pmon -c 1 -s u 2 &> /dev/null | grep "-" | awk '{print $1}' | head -n1
