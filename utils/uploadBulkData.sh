#!/bin/bash
# Utility script to upload data from local file system to Google cloud storage bucket

# For uploading large uncompressed files like directories, use the Google cloud SDK's 'gsutil' utility method with appropriate arguments

gsutil cp -r -m  gs://["Cloud-ML-Data"]/

# if uploading a compressed file to a bucket in Google cloud storage, skip the recursive option

gsutil cp ["CloudML.zip" gs://["Cloud-ML-Data"]/