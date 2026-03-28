## Raabin-det Dataset

This dataset is a refined detection-compatible version of Raabin-WBC. The images can be downloaded from (http://dl.raabindata.com/WBC/Second_microscope/Album_4.zip), and the refined annotation files can be found in `MCDC-YOLO/data/Raabin-det/Raabin_det_test_instance.json`.

## Weights

The trained weights of MCDC-YOLO can be downloaded from (https://1drv.ms/u/c/752d9f191ce33dea/EXYBsTG0gJRInSmFBokmpg0B74Z1jP-xjwdtCupm5eW0cQ?e=a3ctep). Detailed training log can be seen at (https://1drv.ms/u/c/752d9f191ce33dea/Ef_5K5Kr3fdHp-U79eGMlK0BAGWm2FvrnjcrRmc2QMn8iA?e=qutf8g).

## Training

You can train the model using the following code:

```bash
export PYTHONPATH="/root/MCDC-YOLO:$PYTHONPATH"
bash tools/dist_train.sh yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py 8
```

## Evaluation

You can also evaluate the model using our provided weights, you can run the code:

```bash
export PYTHONPATH="/root/MCDC-YOLO:$PYTHONPATH"
bash tools/dist_test.sh yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py contrastive-yolo-world_3e-4.pth 8
```

## Inference

You can perform inference using the following code:

```bash
export PYTHONPATH="/root/MCDC-YOLO:$PYTHONPATH"
python demo/image_demo.py yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py contrastive-yolo-world_3e-4.pth images text_path.txt --output-dir output_path
```

