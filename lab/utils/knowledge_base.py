import json
import boto3
import time
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, RequestError
import pprint
from retrying import retry
import traceback

import warnings
warnings.filterwarnings('ignore')

valid_generation_models = ["amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-micro-v1:0"]
valid_reranking_models = ["cohere.rerank-v3-5:0"] 
valid_embedding_models = ["amazon.titan-embed-text-v2:0", "amazon.titan-embed-image-v1:0"]

embedding_context_dimensions = {
    "amazon.titan-embed-text-v2:0": 1024
}

pp = pprint.PrettyPrinter(indent=2)

def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)

class BedrockKnowledgeBase:
    def __init__(
            self,
            kb_name=None,
            kb_description=None,
            data_sources=None,
            embedding_model="amazon.titan-embed-text-v2:0",
            generation_model="amazon.nova-pro-v1:0",
            reranking_model="cohere.rerank-v3-5:0",
            chunking_strategy="FIXED_SIZE",
            suffix=None,
    ):
        boto3_session = boto3.session.Session()
        self.region_name = boto3_session.region_name
        self.iam_client = boto3_session.client('iam')
        self.account_number = boto3.client('sts').get_caller_identity().get('Account')
        self.suffix = suffix or f'{self.region_name}-{self.account_number}'
        self.identity = boto3.client('sts').get_caller_identity()['Arn']
        self.aoss_client = boto3_session.client('opensearchserverless')
        self.s3_client = boto3.client('s3')
        self.bedrock_agent_client = boto3.client('bedrock-agent')
        credentials = boto3.Session().get_credentials()
        self.awsauth = AWSV4SignerAuth(credentials, self.region_name, 'aoss')

        self.kb_name = kb_name or f"default-knowledge-base-{self.suffix}"
        self.kb_description = kb_description or "Default Knowledge Base"
        self.data_sources = data_sources
        self.bucket_names = [d["bucket_name"] for d in self.data_sources if d['type']== 'S3']
        self.chunking_strategy = chunking_strategy
        
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.reranking_model = reranking_model
        
        self._validate_models()
        
        # Set policy names
        self.encryption_policy_name = f"bedrock-sample-rag-sp-{self.suffix}"
        self.network_policy_name = f"bedrock-sample-rag-np-{self.suffix}"
        self.access_policy_name = f'bedrock-sample-rag-ap-{self.suffix}'
        self.kb_execution_role_name = f'BedrockExecutionRoleForKnowledgeBase_{self.suffix}'
        self.fm_policy_name = f'BedrockFoundationModelPolicyForKnowledgeBase_{self.suffix}'
        self.s3_policy_name = f'BedrockS3PolicyForKnowledgeBase_{self.suffix}'
        self.oss_policy_name = f'BedrockOSSPolicyForKnowledgeBase_{self.suffix}'
        self.bda_policy_name = f'BedrockBDAPolicyForKnowledgeBase_{self.suffix}'

        self.vector_store_name = f'bedrock-sample-rag-{self.suffix}'
        self.index_name = f"bedrock-sample-rag-index-{self.suffix}"

        self._setup_resources()

    def _validate_models(self):
        if self.embedding_model not in valid_embedding_models:
            raise ValueError(f"Invalid embedding model. Your embedding model should be one of {valid_embedding_models}")
        if self.generation_model not in valid_generation_models:
            raise ValueError(f"Invalid Generation model. Your generation model should be one of {valid_generation_models}")
        if self.reranking_model not in valid_reranking_models:
            raise ValueError(f"Invalid Reranking model. Your reranking model should be one of {valid_reranking_models}")

    def _setup_resources(self):
        print("========================================================================================")
        print(f"Step 1 - Creating or retrieving S3 bucket(s) for Knowledge Base documents")
        self.create_s3_bucket()
        
        print("========================================================================================")
        print(f"Step 2 - Creating Knowledge Base Execution Role and Policies")
        self.bedrock_kb_execution_role = self.create_bedrock_execution_role()
        
        print("========================================================================================")
        print(f"Step 3 - Creating OSS encryption, network and data access policies")
        self.encryption_policy, self.network_policy, self.access_policy = self.create_policies_in_oss()
        
        print("========================================================================================")
        print(f"Step 4 - Creating OSS Collection (this step takes a couple of minutes to complete)")
        self.host, self.collection, self.collection_id, self.collection_arn = self.create_oss()
        self.oss_client = OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        
        print("========================================================================================")
        print(f"Step 5 - Creating OSS Vector Index")
        self.create_vector_index()
        
        print("========================================================================================")
        print(f"Step 6 - Creating Knowledge Base")
        self.knowledge_base, self.data_source = self.create_knowledge_base()
        print("========================================================================================")

    def create_s3_bucket(self):
        for bucket_name in self.bucket_names:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                print(f'Bucket {bucket_name} already exists - retrieving it!')
            except ClientError:
                print(f'Creating bucket {bucket_name}')
                if self.region_name == "us-east-1":
                    self.s3_client.create_bucket(Bucket=bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region_name}
                    )

    def create_bedrock_execution_role(self):
        # Create foundation model policy
        foundation_model_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel"],
                    "Resource": [
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.embedding_model}",
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.generation_model}",
                        f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.reranking_model}"
                    ]
                }
            ]
        }

        # Create S3 policy
        s3_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [item for sublist in [[f'arn:aws:s3:::{bucket}', f'arn:aws:s3:::{bucket}/*'] 
                                                     for bucket in self.bucket_names] for item in sublist],
                    "Condition": {
                        "StringEquals": {
                            "aws:ResourceAccount": f"{self.account_number}"
                        }
                    }
                }
            ]
        }

        # Create BDA policy
        bda_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:GetDataAutomationStatus",
                        "bedrock:InvokeDataAutomationAsync"
                    ],
                    "Resource": [
                        f"arn:aws:bedrock:{self.region_name}:{self.account_number}:data-automation-invocation/*",
                        f"arn:aws:bedrock:{self.region_name}:aws:data-automation-project/public-rag-default"
                    ]
                }
            ]
        }

        # Create role
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        try:
            bedrock_kb_execution_role = self.iam_client.create_role(
                RoleName=self.kb_execution_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
            )
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            bedrock_kb_execution_role = self.iam_client.get_role(RoleName=self.kb_execution_role_name)

        # Create and attach policies
        policies = [
            (self.fm_policy_name, foundation_model_policy_document),
            (self.s3_policy_name, s3_policy_document),
            (self.bda_policy_name, bda_policy_document)
        ]

        for policy_name, policy_document in policies:
            try:
                policy = self.iam_client.create_policy(
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(policy_document)
                )
                self.iam_client.attach_role_policy(
                    RoleName=self.kb_execution_role_name,
                    PolicyArn=policy["Policy"]["Arn"]
                )
            except self.iam_client.exceptions.EntityAlreadyExistsException:
                policy_arn = f"arn:aws:iam::{self.account_number}:policy/{policy_name}"
                self.iam_client.attach_role_policy(
                    RoleName=self.kb_execution_role_name,
                    PolicyArn=policy_arn
                )

        return bedrock_kb_execution_role

    def create_policies_in_oss(self):
        try:
            encryption_policy = self.aoss_client.create_security_policy(
                name=self.encryption_policy_name,
                policy=json.dumps(
                    {
                        'Rules': [{'Resource': ['collection/' + self.vector_store_name],
                                   'ResourceType': 'collection'}],
                        'AWSOwnedKey': True
                    }),
                type='encryption'
            )
        except self.aoss_client.exceptions.ConflictException:
            encryption_policy = self.aoss_client.get_security_policy(
                name=self.encryption_policy_name,
                type='encryption'
            )

        try:
            network_policy = self.aoss_client.create_security_policy(
                name=self.network_policy_name,
                policy=json.dumps(
                    [
                        {'Rules': [{'Resource': ['collection/' + self.vector_store_name],
                                    'ResourceType': 'collection'}],
                         'AllowFromPublic': True}
                    ]),
                type='network'
            )
        except self.aoss_client.exceptions.ConflictException:
            network_policy = self.aoss_client.get_security_policy(
                name=self.network_policy_name,
                type='network'
            )

        try:
            access_policy = self.aoss_client.create_access_policy(
                name=self.access_policy_name,
                policy=json.dumps(
                    [
                        {
                            'Rules': [
                                {
                                    'Resource': ['collection/' + self.vector_store_name],
                                    'Permission': [
                                        'aoss:CreateCollectionItems',
                                        'aoss:DeleteCollectionItems',
                                        'aoss:UpdateCollectionItems',
                                        'aoss:DescribeCollectionItems'],
                                    'ResourceType': 'collection'
                                },
                                {
                                    'Resource': ['index/' + self.vector_store_name + '/*'],
                                    'Permission': [
                                        'aoss:CreateIndex',
                                        'aoss:DeleteIndex',
                                        'aoss:UpdateIndex',
                                        'aoss:DescribeIndex',
                                        'aoss:ReadDocument',
                                        'aoss:WriteDocument'],
                                    'ResourceType': 'index'
                                }],
                            'Principal': [self.identity, self.bedrock_kb_execution_role['Role']['Arn']],
                            'Description': 'Data access policy'
                        }
                    ]),
                type='data'
            )
        except self.aoss_client.exceptions.ConflictException:
            access_policy = self.aoss_client.get_access_policy(
                name=self.access_policy_name,
                type='data'
            )

        return encryption_policy, network_policy, access_policy

    def create_oss(self):
        try:
            collection = self.aoss_client.create_collection(
                name=self.vector_store_name, 
                type='VECTORSEARCH'
            )
            collection_id = collection['createCollectionDetail']['id']
            collection_arn = collection['createCollectionDetail']['arn']
        except self.aoss_client.exceptions.ConflictException:
            collection = self.aoss_client.batch_get_collection(
                names=[self.vector_store_name]
            )['collectionDetails'][0]
            collection_id = collection['id']
            collection_arn = collection['arn']

        host = collection_id + '.' + self.region_name + '.aoss.amazonaws.com'

        response = self.aoss_client.batch_get_collection(names=[self.vector_store_name])
        while (response['collectionDetails'][0]['status']) == 'CREATING':
            print('Creating collection...')
            interactive_sleep(30)
            response = self.aoss_client.batch_get_collection(names=[self.vector_store_name])

        try:
            self.create_oss_policy(collection_id)
            print("Sleeping for a minute to ensure data access rules have been enforced")
            interactive_sleep(60)
        except Exception as e:
            print("Policy already exists")
            pp.pprint(e)

        return host, collection, collection_id, collection_arn

    def create_oss_policy(self, collection_id):
        oss_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [f"arn:aws:aoss:{self.region_name}:{self.account_number}:collection/{collection_id}"]
                }
            ]
        }
        try:
            oss_policy = self.iam_client.create_policy(
                PolicyName=self.oss_policy_name,
                PolicyDocument=json.dumps(oss_policy_document),
                Description='Policy for accessing opensearch serverless',
            )
            oss_policy_arn = oss_policy["Policy"]["Arn"]
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            oss_policy_arn = f"arn:aws:iam::{self.account_number}:policy/{self.oss_policy_name}"
        
        self.iam_client.attach_role_policy(
            RoleName=self.bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=oss_policy_arn
        )

    def create_vector_index(self):
        body_json = {
            "settings": {
                "index.knn": "true",
                "number_of_shards": 1,
                "knn.algo_param.ef_search": 512,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": embedding_context_dimensions[self.embedding_model],
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "l2"
                        },
                    },
                    "text": {
                        "type": "text"
                    },
                    "text-metadata": {
                        "type": "text"}
                }
            }
        }

        try:
            response = self.oss_client.indices.create(index=self.index_name, body=json.dumps(body_json))
            print('\nCreating index:')
            pp.pprint(response)
            interactive_sleep(60)
        except RequestError as e:
            print(f'Error while trying to create the index, with error {e.error}')

    def create_chunking_strategy_config(self, strategy):
        configs = {
            "NONE": {
                "chunkingConfiguration": {"chunkingStrategy": "NONE"}
            },
            "FIXED_SIZE": {
                "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 300,
                    "overlapPercentage": 20
                    }
                }
            }
        }
        return configs.get(strategy, configs["NONE"])

    @retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
    def create_knowledge_base(self):
        opensearch_serverless_configuration = {
            "collectionArn": self.collection_arn,
            "vectorIndexName": self.index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata"
            }
        }

        embedding_model_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.embedding_model}"
        knowledgebase_configuration = { 
            "type": "VECTOR", 
            "vectorKnowledgeBaseConfiguration": { 
                "embeddingModelArn": embedding_model_arn
            }
        }
        
        try:
            create_kb_response = self.bedrock_agent_client.create_knowledge_base(
                name=self.kb_name,
                description=self.kb_description,
                roleArn=self.bedrock_kb_execution_role['Role']['Arn'],
                knowledgeBaseConfiguration=knowledgebase_configuration,
                storageConfiguration={
                    "type": "OPENSEARCH_SERVERLESS",
                    "opensearchServerlessConfiguration": opensearch_serverless_configuration
                }
            )
            kb = create_kb_response["knowledgeBase"]
            pp.pprint(kb)
        except self.bedrock_agent_client.exceptions.ConflictException:
            kbs = self.bedrock_agent_client.list_knowledge_bases(maxResults=100)
            kb_id = next((kb['knowledgeBaseId'] for kb in kbs['knowledgeBaseSummaries'] if kb['name'] == self.kb_name), None)
            response = self.bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
            kb = response['knowledgeBase']
            pp.pprint(kb)
          
        # Create Data Sources
        print("Creating Data Sources")
        ds_list = []
        chunking_strategy_configuration = self.create_chunking_strategy_config(self.chunking_strategy)
        
        for idx, ds in enumerate(self.data_sources):
            if ds['type'] == "S3":
                ds_name = f'{kb["knowledgeBaseId"]}-s3'
                s3_data_source_configuration = {
                    "type": "S3",
                    "s3Configuration":{
                        "bucketArn": f'arn:aws:s3:::{ds["bucket_name"]}'
                    }
                }
                
                vector_ingestion_configuration = {
                    "chunkingConfiguration": chunking_strategy_configuration['chunkingConfiguration']
                }

                create_ds_response = self.bedrock_agent_client.create_data_source(
                    name = ds_name,
                    description = self.kb_description,
                    knowledgeBaseId = kb['knowledgeBaseId'],
                    dataSourceConfiguration = s3_data_source_configuration,
                    vectorIngestionConfiguration = vector_ingestion_configuration
                )
                ds = create_ds_response["dataSource"]
                pp.pprint(ds)
                ds_list.append(ds)
                
        return kb, ds_list

    def start_ingestion_job(self):
        for idx, ds in enumerate(self.data_source):
            try:
                start_job_response = self.bedrock_agent_client.start_ingestion_job(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                    dataSourceId=ds["dataSourceId"]
                )
                job = start_job_response["ingestionJob"]
                print(f"job {idx+1} started successfully\n")
                
                while job['status'] not in ["COMPLETE", "FAILED", "STOPPED"]:
                    get_job_response = self.bedrock_agent_client.get_ingestion_job(
                        knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                        dataSourceId=ds["dataSourceId"],
                        ingestionJobId=job["ingestionJobId"]
                    )
                    job = get_job_response["ingestionJob"]
                pp.pprint(job)
                interactive_sleep(40)

            except Exception as e:
                print(f"Couldn't start {idx} job.\n")
                print(e)

    def delete_kb(self, delete_s3_bucket=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Delete data sources
            ds_id_list = self.bedrock_agent_client.list_data_sources(
                knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                maxResults=100
            )['dataSourceSummaries']

            for idx, ds in enumerate(ds_id_list):
                try:
                    self.bedrock_agent_client.delete_data_source(
                        dataSourceId=ds_id_list[idx]["dataSourceId"],
                        knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                    )
                    print("======== Data source deleted =========")
                except Exception as e:
                    print(e)
            
            # Delete KB
            try:
                self.bedrock_agent_client.delete_knowledge_base(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                )
                print("======== Knowledge base deleted =========")
            except Exception as e:
                print(e)

            time.sleep(20)

            # Delete OSS collection and policies
            try:
                self.aoss_client.delete_collection(id=self.collection_id)
                self.aoss_client.delete_access_policy(type="data", name=self.access_policy_name)
                self.aoss_client.delete_security_policy(type="network", name=self.network_policy_name)
                self.aoss_client.delete_security_policy(type="encryption", name=self.encryption_policy_name)
                print("======== Vector Index, collection and associated policies deleted =========")
            except Exception as e:
                print(e)
            
            # Delete role and policies
            self.delete_iam_role_and_policies()

            # Delete S3 bucket if requested
            if delete_s3_bucket:
                for bucket_name in self.bucket_names:
                    try:
                        bucket = boto3.resource('s3').Bucket(bucket_name)
                        bucket.objects.all().delete()
                        bucket.delete()
                        print(f"Deleted bucket {bucket_name}")
                    except Exception as e:
                        print(f"Error deleting bucket {bucket_name}: {e}")

    def delete_iam_role_and_policies(self):
        # Fetch attached policies
        response = self.iam_client.list_attached_role_policies(
            RoleName=self.kb_execution_role_name
        )
        policies_to_detach = response['AttachedPolicies']

        for policy in policies_to_detach:
            policy_arn = policy['PolicyArn']
            try:
                self.iam_client.detach_role_policy(
                    RoleName=self.kb_execution_role_name,
                    PolicyArn=policy_arn
                )
                self.iam_client.delete_policy(PolicyArn=policy_arn)
            except Exception as e:
                print(f"Error detaching/deleting policy {policy_arn}: {e}")

        try:
            self.iam_client.delete_role(RoleName=self.kb_execution_role_name)
            print("======== All IAM roles and policies deleted =========")
        except Exception as e:
            print(f"Error deleting role {self.kb_execution_role_name}: {e}")


    def create_nova_inference_profile(self, profile_name, throughput=1):
        try:
            bedrock_client = boto3.client('bedrock')
            
            request_params = {
                "inferenceProfileName": profile_name,
                "modelSource": {
                    "copyFrom": f"arn:aws:bedrock:{self.region_name}:{self.account_number}:inference-profile/us.{self.generation_model}"
                }
            }
            
            try:
                existing_profiles = bedrock_client.list_inference_profiles()
                for profile in existing_profiles.get('inferenceProfiles', []):
                    if profile['inferenceProfileName'] == profile_name:
                        print(f"Profile {profile_name} already exists")
                        return profile['inferenceProfileArn']
            except Exception as e:
                print(f"Error checking existing profiles: {str(e)}")
        
            response = bedrock_client.create_inference_profile(**request_params)
            
            profile_arn = response['inferenceProfileArn']
            
            max_attempts = 30
            attempt = 0
            while attempt < max_attempts:
                try:
                    status_response = bedrock_client.get_inference_profile(
                        inferenceProfileIdentifier=profile_arn
                    )
                    status = status_response['status']
                    print(f"Current status: {status}")
                    
                    if status == 'ACTIVE':
                        print(f"Inference profile created successfully: {profile_arn}")
                        return profile_arn
                    elif status in ['FAILED', 'DELETED']:
                        raise Exception(f"Profile creation failed with status: {status}")
                    
                    print("Waiting for profile to be ready...")
                    time.sleep(10)
                    attempt += 1
                except Exception as e:
                    print(f"Error checking status: {str(e)}")
                    raise
            
            raise Exception("Profile creation timed out")
            
        except Exception as e:
            print(f"Error creating inference profile: {str(e)}")
            traceback.print_exc()
            raise
    
    def delete_nova_inference_profile(self, profile_name):
        try:
            self.bedrock_agent_client.delete_inference_profile(
                inferenceProfileIdentifier=profile_name
            )
            print(f"Profile {profile_name} deleted successfully")
            
        except Exception as e:
            print(f"Error deleting profile: {str(e)}")
