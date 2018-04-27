
'''Credentials saved to file: [/Users/tejbir/.config/gcloud/application_default_credentials.json]
These credentials will be used by any library that requests
Application Default Credentials.

To generate an access token for other uses, run:
  gcloud auth application-default print-access-token'''

# from google.cloud import datastore

import sys
import os.path
import subprocess
import shlex


def usage():
    print("""
Usage:
        storage createbucket <PROJECT_ID> <STORAGE_CLASS> <BUCKET_LOCATION> <BUCKET_NAME>
        storage deletebucket <PROJECT_ID> <BUCKET_NAME>
        storage listbuckets {..list of buckets..}

        compute createvminstance <PROJECT_ID> <INSTANCE_NAME> <IMAGE_NAME> <ZONE> <MACHINE_TYPE> <DISK_SIZE> 
        compute viewvminstances {..list of vm instances..}
        compute stoprunninginstance <INSTANCE_NAME> <ZONE>
        compute deletevminstance <INSTANCE_NAME>
        
        help [-h] """)


def handlevmrequest(args):
    if (args[2] == 'createvminstance' and len(args) == 9 ):
        print('received command to create vm instance..')
        subprocess.check_call(['./createVMFromDLImage.sh', args[3], args[4], args[5], args[6], args[7], args[8]])
    elif (args[2] == 'stoprunninginstance'):
        print("Stopping VM instance")
        subprocess.check_call(['./stopvminstance.sh', args[3], args[4]])
    elif (args[2] == 'deletevminstance'):
        print("Deleting stopped vm instance")
        subprocess.check_call(['./deletestoppedinstance.sh', args[3], args[4]])
    elif (args[2] == 'viewvminstances'):
        print('Listing all current VM instances')
        subprocess.call(['./listallinstances.sh'])
    else:
        print("Invalid arguments or unsupported GCP operation requested. Try again.")
        usage()
    return 0

def handlestoragerequest(args):
    if (args[2] == 'createbucket' and len(args) == 7):
        print("Received command to create bucket..")
        print("ProjectID: ", args[3])
        print("Default Storage Class: ", args[4])
        print("Bucket location: ", args[5])
        print("Bucket name: ", args[6])
        # subprocess.call(shlex.split('./createbucket.sh args[3] args[4] args[5] args[6]'))
        subprocess.check_call(['./createbucket.sh', args[3], args[4], args[5], args[6]])
    elif (args[2] == 'deletebucket'):
        print('Received command to delete bucket..')
        print('Bucket name: ', args[3])
        subprocess.check_call(['./deletebucketandcontents.sh', args[3]])
    elif (args[2] == 'listbuckets'):
        print('Argument list received from main: ', args)
        print('Received command to list all buckets in current project..')
        subprocess.check_call(['./listbucketsinproject.sh', args[2]])
    else:
        print("Invalid arguments or unsupported GCP operation requested. Try again.")
        usage()
    return 0


def main():
    # [TODO]: make the following a regex so it can handle any variant and mix usage of lower or camel case letters
    if (len(sys.argv) == 1 or sys.argv[1] == 'help' or sys.argv[1] == '-h' ) :
        usage()
    elif (sys.argv[1] == 'compute'):
        handlevmrequest(sys.argv)
    elif (sys.argv[1] == 'storage'):
        handlestoragerequest(sys.argv)
    else:
        print("Invalid arguments or unsupported GCP operation requested. Try again.")
        usage()

main()