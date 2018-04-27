#!/bin/bash
# This script assumes gcloud is installed as aprt of the Google Cloud SDK and is on the user's PATH enviornment variable
# Ths script uses the Google cloud SDK's command line tool to stop a running instance
echo "Received arguments.."
echo "$@"

gcloud compute instances stop $1 --zone=$2