# Creating virtual environments

## Conda
Install [mamba](https://github.com/mxochicale/code/tree/main/mamba) 

## Create virtual environment
### mamba
```
mamba update -n base mamba
mamba create -n VE python=3.8 pip -c conda-forge
mamba activate VE
```

### conda
* [ve.yml](ve.yml)


## Dependencies

* Python package versions
```
$ cd $HOME/.../dependencies
$ mamba activate ve
$ python package_versions.py 


python: 3.11.3 | packaged by conda-forge | (main, Apr  6 2023, 08:57:19) [GCC 11.3.0]
torch: 2.0.1
torchvision: 0.15.2
torch cuda_is_available: True
torch cuda version: 11.8
torch cuda.device_count  1
PIL: 9.5.0


```

* OS
```
$ hostnamectl

 Static hostname: --
       Icon name: computer-laptop
         Chassis: laptop
      Machine ID: --
         Boot ID: --
Operating System: Ubuntu 22.04.1 LTS              
          Kernel: Linux 5.15.0-56-generic
    Architecture: x86-64
 Hardware Vendor: --

```

* GPU
```
$ nvidia-smi -q

==============NVSMI LOG==============

Timestamp                                 : Sat Dec 17 13:27:52 2022
Driver Version                            : 520.61.05
CUDA Version                              : 11.8

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : NVIDIA RTX A2000 8GB Laptop GPU
    Product Brand                         : NVIDIA RTX
    Product Architecture                  : Ampere

```


