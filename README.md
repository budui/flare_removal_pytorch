
## Usage

### training

```bash
python3 main.py ./configs/original_paper.yml --gpus 0
# change loss weights in original_paper.yml
python3 main.py ./configs/original_paper.yml --gpus 0 name="ffl_1-lpips_1|l1_1-lpips_1" \
    loss.weight.flare.ffl=100 \
    loss.weight.flare.l1=0 \
    loss.weight.flare.lpips=1 \
    loss.weight.flare.perceptual=0 \
    loss.weight.scene.ffl=0 \
    loss.weight.scene.l1=1 \
    loss.weight.scene.lpips=1 \
    loss.weight.scene.perceptual=0 \
```

### evaluate using paired images

```bash
python3 test.py evaluate </path/to/config.yml> </path/to/checkpoint>
```

### generate non-flare images

```bash
python3 test.py generate </path/to/config.yml> </path/to/checkpoint> \
    --image_folder </path/to/images-with-flare/> \
    --output_folder </path/to/pred-images/>
```