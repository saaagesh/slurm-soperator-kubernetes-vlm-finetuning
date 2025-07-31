#!/usr/bin/env python3
"""
Create MLflow experiment using HTTP calls
"""
import os
import requests
import json
import time
from requests.auth import HTTPBasicAuth

def create_experiment():
    """Create MLflow experiment using direct HTTP calls"""
    
    base_url = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "vlm-finetuning")
    
    print("=== Creating MLflow Experiment ===")
    print(f"Server: {base_url}")
    print(f"Experiment: {experiment_name}")
    
    # Setup authentication and headers
    auth = HTTPBasicAuth(username, password)
    headers = {'Content-Type': 'application/json'}
    
    try:
        # Step 1: Check if experiment already exists
        print("1. Checking existing experiments...")
        response = requests.get(
            f"{base_url}/api/2.0/mlflow/experiments/search?max_results=100",
            auth=auth,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"   ✗ Failed to list experiments: {response.status_code}")
            return False
        
        experiments = response.json()['experiments']
        existing_exp = None
        
        for exp in experiments:
            if exp['name'] == experiment_name:
                existing_exp = exp
                break
        
        if existing_exp:
            print(f"   ✓ Experiment '{experiment_name}' already exists!")
            print(f"     ID: {existing_exp['experiment_id']}")
            print(f"     Location: {existing_exp['artifact_location']}")
            print(f"     Status: {existing_exp['lifecycle_stage']}")
            return True
        
        # Step 2: Create new experiment
        print(f"2. Creating experiment '{experiment_name}'...")
        
        create_data = {
            "name": experiment_name,
            "tags": [
                {"key": "project", "value": "vlm-finetuning"},
                {"key": "created_by", "value": "http-client"},
                {"key": "created_at", "value": str(int(time.time()))}
            ]
        }
        
        response = requests.post(
            f"{base_url}/api/2.0/mlflow/experiments/create",
            auth=auth,
            headers=headers,
            json=create_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            experiment_id = result['experiment_id']
            print(f"   ✓ Successfully created experiment!")
            print(f"     Name: {experiment_name}")
            print(f"     ID: {experiment_id}")
            
            # Step 3: Verify experiment was created
            print("3. Verifying experiment...")
            verify_response = requests.get(
                f"{base_url}/api/2.0/mlflow/experiments/get?experiment_id={experiment_id}",
                auth=auth,
                timeout=10
            )
            
            if verify_response.status_code == 200:
                exp_info = verify_response.json()['experiment']
                print(f"   ✓ Verification successful!")
                print(f"     Artifact location: {exp_info['artifact_location']}")
                print(f"     Status: {exp_info['lifecycle_stage']}")
                
                # Step 4: Create a test run to fully verify
                print("4. Creating test run...")
                run_data = {
                    "experiment_id": experiment_id,
                    "start_time": int(time.time() * 1000),
                    "tags": [
                        {"key": "run_type", "value": "test"},
                        {"key": "test_phase", "value": "setup"}
                    ]
                }
                
                run_response = requests.post(
                    f"{base_url}/api/2.0/mlflow/runs/create",
                    auth=auth,
                    headers=headers,
                    json=run_data,
                    timeout=10
                )
                
                if run_response.status_code == 200:
                    run_info = run_response.json()['run']['info']
                    run_id = run_info['run_id']
                    print(f"   ✓ Test run created: {run_id}")
                    
                    # Log test parameters and metrics
                    print("5. Logging test data...")
                    
                    # Log parameter
                    param_data = {
                        "run_id": run_id,
                        "key": "test_param",
                        "value": "setup_verification"
                    }
                    requests.post(
                        f"{base_url}/api/2.0/mlflow/runs/log-parameter",
                        auth=auth,
                        headers=headers,
                        json=param_data,
                        timeout=10
                    )
                    
                    # Log metric
                    metric_data = {
                        "run_id": run_id,
                        "key": "setup_success",
                        "value": 1.0,
                        "timestamp": int(time.time() * 1000)
                    }
                    requests.post(
                        f"{base_url}/api/2.0/mlflow/runs/log-metric",
                        auth=auth,
                        headers=headers,
                        json=metric_data,
                        timeout=10
                    )
                    
                    # End the test run
                    end_data = {
                        "run_id": run_id,
                        "status": "FINISHED",
                        "end_time": int(time.time() * 1000)
                    }
                    requests.post(
                        f"{base_url}/api/2.0/mlflow/runs/update",
                        auth=auth,
                        headers=headers,
                        json=end_data,
                        timeout=10
                    )
                    
                    print("   ✓ Test run completed successfully!")
                    
                else:
                    print(f"   ✗ Test run creation failed: {run_response.status_code}")
                    print(f"     Response: {run_response.text}")
                
            else:
                print(f"   ✗ Verification failed: {verify_response.status_code}")
                
            return True
            
        else:
            print(f"   ✗ Failed to create experiment: {response.status_code}")
            print(f"     Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error creating experiment: {e}")
        return False

if __name__ == "__main__":
    success = create_experiment()
    if success:
        print("\n=== EXPERIMENT SETUP COMPLETE ===")
        print("Your MLflow experiment is ready for training!")
        print(f"You can view it at: {os.environ.get('MLFLOW_TRACKING_URI')}")
    else:
        print("\n=== EXPERIMENT SETUP FAILED ===")
    exit(0 if success else 1)