# mimic_mjlab

train g1 mimic in mjlab

## Installation

- ### Dependencies

    ```bash
    sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
    ```

- ### Create a New Environment

    Use the following command to create a virtual environment:

    ```bash
    conda create -n mimic_mjlab python=3.11
    ```

    Activate the Virtual Environment

    ```bash
    conda activate mimic_mjlab
    ```

- ### Clone the Project

    Clone the repository using Git:

    ```bash
    git clone https://github.com/ak1raljl/mimic_mjlab.git
    cd mimic_mjlab
    pip install -e .
    ```

- ### then follow the training steps: prepare motion data -> csv2npz -> train -> play.


## Prepare Motion Files

Convert CSV motion data to NPZ format:

```bash
python scripts/csv2npz.py --input-file src/assets/motions/<robot>/<motion>.csv --output-name <motion>.npz --input-fps <input-fps> --output-fps 30 --robot <robot>

# example
python scripts/csv2npz.py --input-file src/assets/motions/g1/dance1_subject2.csv --output-name dance1_subject2.npz --input-fps 30 --output-fps 50 --robot g1
```

## Verify Motion Files

Playback and validate the converted NPZ files:

```bash
cd train_mimic

# Basic playback with loop
python scripts/play_motion.py --robot g1 --motion-file src/assets/motions/g1/dance1_subject2.npz

# Single playthrough without loop
python scripts/play_motion.py --robot g1 --motion-file src/assets/motions/g1/dance1_subject2.npz --no-loop

# Use web viewer for remote servers
python scripts/play_motion.py --robot g1 --motion-file src/assets/motions/g1/dance1_subject2.npz --viewer viser
```

## Train

```bash
# train
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/<motion>.npz --env.scene.num-envs=4096

# resume training
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/<motion>.npz --env.scene.num-envs=4096 --agent.resume True --agent.load-run 2026-xx-xx_xx-xx-xx --agent.load-checkpoint <model>.pt --agent.max-iterations 30000 --agent.run-name resume
```


## Play

```bash
# play
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation  --motion_file=src/assets/motions/g1/<motion>.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```
