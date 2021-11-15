export PYTHONPATH=/home/azureuser/work/cd_expts/tf_obj_det/research/slim:/home/azureuser/work/cd_expts/tf_obj_det/research
python /home/azureuser/work/cd_expts/tf_obj_det/research/object_detection/model_main_tf2.py --pipeline_config_path=/home/azureuser/work/cd_expts/tf_obj_det/research/object_detection/ts_space/configs/faster_rcnn_resnet101_v1_1024x1024_ts_cotton_tpu-8.config --model_dir=/mnt/ts-cvml-datastore/ts_model/models/cotton --checkpoint_every_n 1000 --wb_group FRC_NV1_009
#python3 object_detection/model_main.py --pipeline_config_path=/home/ubuntu/object_detection_api/train_config/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --model_dir=/home/ubuntu/object_detection_api/detection_benchmark/
#sudo poweroff
