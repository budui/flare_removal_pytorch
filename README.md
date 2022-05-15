
## Usage

### training

```bash
python3 main.py ./configs/original_paper.yml --gpus 0
```

### evaluate using paired images

```bash
python3 test.py evaluate </path/to/config.yml> </path/to/checkpoint>
```

### generate non-flare images

```bash
python3 test.py generate </path/to/config.yml> \
    </path/to/checkpoint> \
    --image_folder </path/to/images-with-flare/> \
    --output_folder </path/to/pred-images/>
```