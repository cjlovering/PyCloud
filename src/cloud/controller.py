import boto3
from typing import List

def allocate(ami_id: str='ami-d874e0a0'):
    """Allocate an instance of the given machine type.
    """
    ec2 = boto3.resource('ec2')
    ec2.create_instances(ImageId=ami_id, MinCount=1, MaxCount=1)
    
def start(instance_ids: List[str]=['i-0d4a2ef9e68a1f847']):
    """Start instances using the given instance_ids.
    """
    client = boto3.client('ec2')
    responses = client.start_instances(
        InstanceIds=[
            instance_ids
        ]
    )

def stop(instance_ids: List[str]=['i-0d4a2ef9e68a1f847']):
    """Stop all instance of the given ids.
    """
    ec2 = boto3.resource('ec2')
    ec2.instances.filter(InstanceIds=instance_ids).stop()
    
def status_check():
    """Print the status of all instances.
    """
    ec2 = boto3.resource('ec2')
    for instance in ec2.instances.all():
        print(instance.id, instance.state)
