Google Cloud VM & Storage instantiation command-line tool

Purpose:  The goal of this script is to facilitate the instantiation of the various configuration of the VMs from the command line. While these scripts use the Google Cloud SDK, they provide a more simple and straightforward API to interact 
with Google’s compute and storage service APIs. This work is pertinent to our project because the focus of our work was training machine learning models on different VM configurations on Google cloud platform while making it easy to tune the vCPUs, RAM, and disk size. Moreover, our scripts provide a way to create VM instances based on a preset custom VM image that we built from an online resource (Please see http://cs231n.github.io/gce-tutorial/).

Where to find the library
The main library driver function and the associated scripts can be found by navigating the following git repo:
https://github.com/cjlovering/PyCloud/tree/master/utils/src

How to use the tool 

Prerequisite #1: 	Ensure that you have python 3.6 installed and that the python interpreter is on your PATH.
Prerequisite #2: 	Ensure that all bash scripts have been given permissions to execute on your local system. By default, bash scripts are not allowed write and execute permissions due to security reasons. 
Prerequisite #3:	Ensure that you’ve initialized the Google SDK environment on your client machine by running the command ‘gcloud init’ and authentication with Google’s authentication service. 
Prerequisite #4:	Ensure that default region and zone have been preconfigured on console.cloud.google.com
Prerequisite #5:	The script assumes that you’ve already setup your Google cloud account and created a project along with a service account, which is required to create VM instances. 

N.B: The tool currently supports two main Google services: storage and compute. Within each service, a number of common operations are supported for each service. 

— For help with the tool, run the main python file, gcpcommonlib.py [help | -h]

— Note: You must run the tool on the command line by supplying the python keyword before the python script. For future improvement, this tool can be made to work without such restriction. 
     Run as: python gcpcommonlib.py -h


Known issues

—The client library assumes that you’ve already installed the latest version of Google cloud SDK on your system and that the key SDK command-line utilities are in your environment’s PATH. 
— The library currently only supports for the default google service account and needs to be tweaked to enable it work with other user service accounts. 
— The script currently only supports VMs with SSD boot disks but this isn’t a limitation and the script can be easily modified to make it more general. 
