# Install Evo2 on EIDF

1. Build container with Kubernetes
2. Install dependencies inside the container
3. Test evo2

## Build container with Kubernetes

Create a container requiring an `NVIDIA-H200` GPU, with sufficient `cpu` cores and `memory`. Use the provided [`job-evo2-run.yaml`](job-evo2-run.yaml). The yaml specifies the following Docker image:

```
nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

Create the container:

```
kubectl create -f job-evo2-run.yaml

# check if container is running
kubectl get po

# if running, switch into container
kubectl exec -it tklaisoo-eidf107-job-n4qgs-49cs6 -- bash
```

## Install dependencies inside the container

Execute the [`install-evo2.sh`](install-evo2.sh) script to install dependencies and clone the evo2 repository. The same script includes the following commands to test the two models.

## Test evo2

The following commands test the two models `evo2_7b` and `evo2_40b` in the environment:
```
python ./test/test_evo2.py --model_name evo2_7b
python ./test/test_evo2.py --model_name evo2_40b
```

The expected command line output to these tests can be found in the following text files:
- [`evo2_7b_test.txt`](evo2_7b_test.txt)
- [`evo2_40b_test.txt`](evo2_40b_test.txt)
