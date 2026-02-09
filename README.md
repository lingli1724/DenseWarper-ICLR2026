# [ICLR Article] From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper

## Resources

Paper: [(arXiv:??)](https://arxiv.org) coming soon！

*MPI-INF-3DHP* Dataset: [(3dhp-dataset)](https://vcai.mpi-inf.mpg.de/3dhp-dataset/)

## Install & Data Preparation

Clone this repo, and install the dependencies.

```
cd DenseWarper
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

The `DenseWarper` directory will be referred as {POSE_ROOT}.

### Compile external modules

```
cd ${POSE_ROOT}/lib/deform_conv
python setup.py develop
```

### Pretrained Models
Download pretrained models. Please download them under {POSE_ROOT}

The pretrained model on *Human3.6M* can be downloaded from this [link](https://drive.google.com/file/d/1VRiB-4DgmVfL7FXRdsWITFDzECrCPwWg/view?usp=drive_link) and the pretrained model on *MPI-INF-3DHP* can be downloaded from this [link](https://drive.google.com/file/d/11tWJXBPC8wSywHISLQpZaV3ayWWvH6ju/view?usp=drive_link)

### Human3.6M
Please follow [CHUNYUWANG/H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox) to prepare the data.

> Note that we have **NO** permission to redistribute the Human3.6M data. Please do NOT ask us for a copy of Human3.6M dataset.

### MPI-INF-3DHP
Please prepare the data following a similar treatment as Human3.6M

## Evaluate
Make sure you are in the {POSE_ROOT} directory.

**Human3.6M**

```bash
python run/DenseWarper/DenseWarper_main.py --cfg experiments/h36m/h36m_4view.yaml --evaluate true
```

**MPI-INF-3DHP**

```bash
python run/DenseWarper/DenseWarper_3dhp_main.py --cfg experiments/3dhp/3dhp_4view.yaml --evaluate true
```

## Results
**Human3.6M**

```
MPJPE summary: j3d_DenseWarper 22.3
```

**MPI-INF-3DHP**

```
MPJPE summary: j3d_DenseWarper 47.89
```

## Train
**Human3.6M**

```bash
python run/DenseWarper/DenseWarper_main.py --cfg experiments/h36m/h36m_4view.yaml --runMode train
```

**MPI-INF-3DHP**

```bash
python run/DenseWarper/DenseWarper_3dhp_main.py --cfg experiments/3dhp/3dhp_4view.yaml --runMode train

```
