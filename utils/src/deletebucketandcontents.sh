#!/bin/bash

# The GSUTIL from the Google cloud SDK is used to create buckets for the currently configured project.
# Ensure that you've first run "gcloud init" to initialize your environment and specified your default project
# gsutil mb -p [PROJECT_NAME] -c [STORAGE_CLASS] -l [BUCKET_LOCATION] gs://[BUCKET_NAME]/

echo "Deleting bucket.."
echo "Received arguments.."
echo "$@"

# gsutil rm -r gs://[BUCKET_NAME]
gsutil rm -r gs://$1