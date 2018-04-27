#!/bin/bash
# This script assumes gcloud is installed as aprt of the Google Cloud SDK and is on the user's PATH enviornment variable
# Ths script uses the Google cloud SDK's command line tool to list all the instances associated with the current project

gcloud compute instances list