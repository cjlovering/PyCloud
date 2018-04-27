#!/bin/bash
# This script assumes gcloud is installed as aprt of the Google Cloud SDK and is on the user's PATH enviornment variable
# Ths script uses the Google cloud SDK's command line tool to instantiate VM instances based off of a Deep learning disk that we leveraged from http://cs231n.github.io/gce-tutorial/
echo "Creating VM instance.."
echo "Received arguments.."
echo "$@"
gcloud compute --project $1 instances create $2 --image $3 --image-project=$1 --zone=$4 --machine-type=$5 --subnet=default --maintenance-policy=MIGRATE --service-account=485656763584-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --min-cpu-platform=Automatic --tags=http-server,https-server --boot-disk-size=$6 --no-boot-disk-auto-delete --boot-disk-type=pd-ssd --boot-disk-device-name=$3