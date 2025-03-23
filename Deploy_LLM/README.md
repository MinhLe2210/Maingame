# 1. Setting up the environment


## 1.1. Installing Docker
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg - dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

## 1.2. Installing NVIDIA Container Toolkit
``` bash
# Configure the production repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg - dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
 && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
 sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
 sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# Optionally, configure the repository to use experimental packages:
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
``` 
##  1.3. Installing kubectl, Helm, and Minikube
###  1.3.1. Execute the scripts install-kubectl.sh, install-helm.sh, and install-minikube-cluster.sh:
``` bash
# Download kubectl and place it in your PATH
bash install-kubectl.sh
# Install kubernetes package manager and add it to your PATH
bash install-helm.sh
# Configure the system to support GPU workloads by enabling the NVIDIA Container Toolkit and starting Minikube with GPU support 
bash install-minikube-cluster.sh
```
### 1.3.2. Verify kubectl installation
``` bash
kubectl version - client
```
### 1.3.3. Verification message of Helm using:
``` bash 
helm version
```

### 1.3.4. Ensure Minikube is running:
``` bash
sudo minikube status
```
### 1.3.5. 
``` bash
sudo kubectl describe nodes | grep -i gpu
```
## 1.4. Verify everything with test workload for GPU
sudo kubectl run gpu-test - image=nvidia/cuda:12.2.0-runtime-ubuntu22.04 - restart=Never - nvidia-smi

Wait until the container is downloaded and the pod is running, to check you can run
``` bash
watch sudo kubectl get pods
``` 
Once the pod is in running stage, get the logs and you should see nvidia-smi output
``` bash
sudo kubectl logs gpu-test
```

# 2. Deploying a single replica
To run your deployment:
``` bash
# Register the helm repo
sudo helm repo add vllm https://vllm-project.github.io/production-stack
# Install the release
sudo helm install vllm vllm/vllm-stack -f assets/values-02-basic-config.yaml
```
## 2.1. To track the status of your newly created pods:
``` bash
watch sudo kubectl get pods
```
Or view more detailed logs and status, here add the complete name of the pod with prefix vllm-llama3-deployment-vllm-xxxxxxxxxx-xxxx:
``` bash
sudo kubectl describe pod <POD_NAME>
```
## 2.2. Test the deployment
``` bash
sudo kubectl port-forward svc/vllm-router-service 30080:80
```

List the deployed models:
```
curl -o- http://localhost:30080/models
```
Send a query to the OpenAI /completion endpoint to generate a completion for a prompt:
``` bash
curl -X POST http://localhost:30080/completions \
 -H "Content-Type: application/json" \
 -d '{
 "model": "meta-llama/Llama-3.1–8B-Instruct",
 "prompt": "Once upon a time,",
 "max_tokens": 100
 }'
 ```

