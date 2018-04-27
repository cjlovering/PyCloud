#!/bin/bash
# This script assumes gcloud is installed as aprt of the Google Cloud SDK and is on the user's PATH enviornment variable
# Ths script uses the Google cloud SDK's command line tool to delete a previously stopped vm instance

echo "Received arguments.."
echo "$@"

gcloud compute instances delete $1 --zone=$2