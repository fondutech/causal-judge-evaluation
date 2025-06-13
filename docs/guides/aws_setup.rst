AWS Integration Guide
====================

Scale up CJE with AWS cloud services for secrets management, distributed processing, and production deployment.

Overview
--------

AWS integration provides:

- **Secrets Management**: Secure API key storage with AWS Secrets Manager
- **Distributed Processing**: Scale evaluations across multiple instances
- **Cost Optimization**: Spot instances and auto-scaling for large workloads
- **Production Deployment**: CI/CD integration and monitoring
- **Data Storage**: S3 integration for datasets and results

Prerequisites
-------------

AWS Account Setup
~~~~~~~~~~~~~~~~~~

1. **AWS Account**: Create an AWS account with appropriate permissions
2. **AWS CLI**: Install and configure AWS CLI:

   .. code-block:: bash

      pip install awscli
      aws configure

3. **IAM Permissions**: Ensure your user/role has the following permissions:

   - ``secretsmanager:GetSecretValue``
   - ``secretsmanager:CreateSecret`` (for setup)
   - ``s3:GetObject``, ``s3:PutObject`` (for data storage)
   - ``ec2:*`` (for distributed processing)

Dependencies
~~~~~~~~~~~~

Install AWS-related dependencies:

.. code-block:: bash

   pip install boto3 awscli
   # or
   poetry add boto3

Secrets Management
------------------

Setting Up AWS Secrets Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Store your API keys securely in AWS Secrets Manager:

.. code-block:: bash

   # Create secret for OpenAI
   aws secretsmanager create-secret \
       --name "cje/api-keys" \
       --description "API keys for CJE evaluation" \
       --secret-string '{
           "OPENAI_API_KEY": "sk-your-openai-key-here",
           "ANTHROPIC_API_KEY": "your-anthropic-key-here",
           "GOOGLE_API_KEY": "your-google-key-here"
       }'

Automatic Secret Retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CJE can automatically fetch API keys from AWS Secrets Manager:

.. code-block:: python

   from cje.utils.aws_secrets import setup_api_keys_from_secrets

   # Automatically setup all API keys
   setup_api_keys_from_secrets(
       secret_name="cje/api-keys",
       region_name="us-east-1"
   )

   # Keys are now available as environment variables
   import os
   print(os.getenv("OPENAI_API_KEY"))  # Automatically populated

Configuration
~~~~~~~~~~~~~

Configure AWS secrets in your CJE configuration:

.. code-block:: yaml

   aws:
     secrets:
       secret_name: "cje/api-keys"
       region_name: "us-east-1"
       
   # Alternative: specify individual secrets
   aws:
     secrets:
       openai_secret: "cje/openai-key"
       anthropic_secret: "cje/anthropic-key"

Manual Secret Retrieval
~~~~~~~~~~~~~~~~~~~~~~~

For more control over secret management:

.. code-block:: python

   from cje.utils.aws_secrets import get_api_key_from_secrets

   # Get specific API key
   openai_key = get_api_key_from_secrets(
       secret_name="cje/api-keys",
       key_name="OPENAI_API_KEY",
       region_name="us-east-1"
   )

   # Set environment variable manually
   import os
   os.environ["OPENAI_API_KEY"] = openai_key

Error Handling
~~~~~~~~~~~~~~

Robust error handling for production:

.. code-block:: python

   from cje.utils.aws_secrets import setup_api_keys_from_secrets
   from botocore.exceptions import ClientError

   try:
       setup_api_keys_from_secrets("cje/api-keys")
       print("✅ API keys loaded from AWS Secrets Manager")
   except ClientError as e:
       error_code = e.response['Error']['Code']
       if error_code == 'ResourceNotFoundException':
           print("❌ Secret not found. Please create it first.")
       elif error_code == 'AccessDeniedException':
           print("❌ Access denied. Check IAM permissions.")
       else:
           print(f"❌ AWS error: {e}")
   except Exception as e:
       print(f"❌ Unexpected error: {e}")
       # Fallback to local environment variables
       print("Using local environment variables as fallback")

S3 Data Storage
---------------

Dataset Storage
~~~~~~~~~~~~~~~

Store large datasets in S3 for efficient access:

.. code-block:: python

   from cje.data.s3 import S3Dataset

   # Load dataset from S3
   dataset = S3Dataset(
       bucket="my-cje-bucket",
       key="datasets/arena_data.jsonl",
       region="us-east-1"
   )

   # Use in configuration
   dataset_config = {
       "name": "S3Dataset",
       "bucket": "my-cje-bucket", 
       "key": "datasets/arena_data.jsonl"
   }

Results Storage
~~~~~~~~~~~~~~~

Automatically save results to S3:

.. code-block:: yaml

   paths:
     work_dir: "s3://my-cje-bucket/results/"
     
   # Or configure specific outputs
   outputs:
     save_to_s3: true
     s3_bucket: "my-cje-bucket"
     s3_prefix: "experiments/"

Batch Processing
~~~~~~~~~~~~~~~~

Process large datasets using S3:

.. code-block:: python

   from cje.aws.batch import S3BatchProcessor

   processor = S3BatchProcessor(
       input_bucket="input-data-bucket",
       output_bucket="results-bucket",
       batch_size=1000
   )

   # Process all files in bucket
   processor.process_bucket("datasets/")

Distributed Processing
----------------------

EC2 Cluster Setup
~~~~~~~~~~~~~~~~~

Launch distributed evaluation cluster:

.. code-block:: python

   from cje.aws.cluster import CJECluster

   cluster = CJECluster(
       instance_type="c5.2xlarge",
       num_instances=5,
       spot_instances=True,  # Cost optimization
       region="us-east-1"
   )

   # Launch cluster
   cluster.launch()

   # Run distributed evaluation
   results = cluster.run_evaluation(
       config_path="config/large_scale_eval.yaml",
       dataset_splits=["split_1", "split_2", "split_3", "split_4", "split_5"]
   )

   # Cleanup
   cluster.terminate()

Auto-Scaling Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   aws:
     cluster:
       auto_scaling:
         min_instances: 1
         max_instances: 20
         target_cpu_utilization: 70
         scale_up_cooldown: 300    # seconds
         scale_down_cooldown: 600  # seconds
         
       instance_config:
         instance_type: "c5.large"
         spot_instances: true
         spot_max_price: "0.10"    # USD per hour

Parallel Arena Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Distribute arena analysis across multiple instances:

.. code-block:: bash

   # Launch distributed arena analysis
   python scripts/run_distributed_arena.py \
       --cluster-size 10 \
       --max-samples 50000 \
       --instance-type c5.2xlarge \
       --spot-instances

Cost Optimization
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use spot instances for cost savings
   cluster_config = {
       "instance_type": "c5.2xlarge",
       "spot_instances": True,
       "spot_max_price": "0.15",  # 70% savings vs on-demand
       
       # Automatic instance termination
       "max_runtime_hours": 4,
       "terminate_on_completion": True,
       
       # Mixed instance types for availability
       "instance_types": ["c5.2xlarge", "c5.4xlarge", "m5.2xlarge"]
   }

Production Deployment
---------------------

CI/CD Integration
~~~~~~~~~~~~~~~~~

GitHub Actions workflow for automated evaluation:

.. code-block:: yaml

   # .github/workflows/cje-evaluation.yml
   name: CJE Evaluation Pipeline
   
   on:
     push:
       paths: ['configs/**', 'data/**']
     schedule:
       - cron: '0 8 * * *'  # Daily at 8 AM
   
   jobs:
     evaluate:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         
         - name: Configure AWS credentials
           uses: aws-actions/configure-aws-credentials@v2
           with:
             aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
             aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
             aws-region: us-east-1
             
         - name: Setup Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.9'
             
         - name: Install CJE
           run: |
             pip install -e .
             
         - name: Run evaluation
           run: |
             python scripts/run_arena_analysis.py \
               --config-from-s3 s3://my-config-bucket/prod-config.yaml \
               --output-to-s3 s3://my-results-bucket/$(date +%Y-%m-%d)/
               
         - name: Upload results
           run: |
             aws s3 sync outputs/ s3://my-results-bucket/github-actions/

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Production environment setup:

.. code-block:: bash

   # Production secrets
   aws secretsmanager create-secret \
       --name "cje/prod/api-keys" \
       --description "Production API keys for CJE" \
       --secret-string '{
           "OPENAI_API_KEY": "sk-prod-key",
           "ANTHROPIC_API_KEY": "prod-anthropic-key"
       }'

   # Staging secrets  
   aws secretsmanager create-secret \
       --name "cje/staging/api-keys" \
       --description "Staging API keys for CJE" \
       --secret-string '{
           "OPENAI_API_KEY": "sk-staging-key",
           "ANTHROPIC_API_KEY": "staging-anthropic-key"
       }'

Lambda Functions
~~~~~~~~~~~~~~~~

Serverless evaluation for lightweight tasks:

.. code-block:: python

   # lambda_function.py
   import json
   from cje.utils.aws_secrets import setup_api_keys_from_secrets
   from cje.config import simple_config
   from cje.pipeline import run_pipeline

   def lambda_handler(event, context):
       # Setup API keys from secrets
       setup_api_keys_from_secrets("cje/prod/api-keys")
       
       # Get configuration from event
       config = simple_config(
           dataset_name=event['dataset_s3_path'],
           estimator_name=event.get('estimator', 'DRCPO')
       )
       
       # Run evaluation
       results = run_pipeline(config)
       
       # Return results
       return {
           'statusCode': 200,
           'body': json.dumps({
               'estimates': results.estimates.tolist(),
               'standard_errors': results.standard_errors.tolist()
           })
       }

Monitoring and Logging
----------------------

CloudWatch Integration
~~~~~~~~~~~~~~~~~~~~~~

Monitor CJE evaluations with CloudWatch:

.. code-block:: python

   from cje.aws.monitoring import CloudWatchLogger

   # Setup monitoring
   logger = CloudWatchLogger(
       log_group="/cje/evaluations",
       metrics_namespace="CJE/Evaluations"
   )

   # Log evaluation metrics
   logger.log_metric("EstimationAccuracy", 0.95)
   logger.log_metric("ProcessingTime", 120.5, unit="Seconds")
   logger.log_metric("APICost", 25.30, unit="Count")

   # Log events
   logger.log_event("evaluation_started", {
       "config": "arena_test.yaml",
       "samples": 1000,
       "policies": 5
   })

Alerts and Notifications
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # CloudWatch Alarms
   aws:
     monitoring:
       alerts:
         - name: "HighAPIError"
           metric: "CJE/Evaluations/ErrorRate"
           threshold: 0.05
           comparison: "GreaterThanThreshold"
           sns_topic: "arn:aws:sns:us-east-1:123456789:cje-alerts"
           
         - name: "HighCost"
           metric: "CJE/Evaluations/APICost"
           threshold: 100.0
           comparison: "GreaterThanThreshold"
           sns_topic: "arn:aws:sns:us-east-1:123456789:cje-cost-alerts"

Security Best Practices
-----------------------

IAM Roles and Policies
~~~~~~~~~~~~~~~~~~~~~~

Create least-privilege IAM policies:

.. code-block:: json

   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "secretsmanager:GetSecretValue"
         ],
         "Resource": "arn:aws:secretsmanager:*:*:secret:cje/*"
       },
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:PutObject"
         ],
         "Resource": [
           "arn:aws:s3:::my-cje-bucket/*"
         ]
       }
     ]
   }

Secret Rotation
~~~~~~~~~~~~~~~

Implement automatic secret rotation:

.. code-block:: python

   from cje.aws.security import SecretRotator

   rotator = SecretRotator(
       secret_name="cje/api-keys",
       rotation_interval_days=30
   )

   # Setup automatic rotation
   rotator.enable_automatic_rotation()

   # Manual rotation
   rotator.rotate_secret("OPENAI_API_KEY", new_key)

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Secret Not Found**

.. code-block:: text

   botocore.exceptions.ClientError: Secret not found

**Solutions:**

- Verify secret name and region
- Check IAM permissions
- Ensure secret exists in the correct region

**Access Denied**

.. code-block:: text

   botocore.exceptions.ClientError: Access denied

**Solutions:**

- Check IAM permissions for your user/role
- Verify resource ARNs in policies
- Ensure you're using the correct AWS credentials

**High Costs**

**Monitoring:**

- Use CloudWatch billing alerts
- Monitor API usage metrics
- Set up cost budgets and alerts

**Cost Reduction:**

- Use spot instances for batch processing
- Implement request caching
- Optimize API usage patterns

Best Practices
--------------

**Security:**

- Use IAM roles instead of access keys when possible
- Implement least-privilege access policies
- Enable CloudTrail for audit logging
- Rotate secrets regularly

**Cost Management:**

- Use spot instances for non-critical workloads
- Implement auto-scaling to match demand
- Monitor and alert on cost thresholds
- Use S3 lifecycle policies for data retention

**Reliability:**

- Implement retry logic with exponential backoff
- Use multiple availability zones
- Set up monitoring and alerting
- Test disaster recovery procedures

**Performance:**

- Use appropriate instance types for workload
- Implement connection pooling
- Cache frequently accessed data
- Optimize data transfer patterns

This AWS integration enables enterprise-scale CJE deployments with proper security, monitoring, and cost management. 