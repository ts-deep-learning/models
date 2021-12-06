export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=/home/azureuser/work/cd_expts/tf_obj_det/research/slim:/home/azureuser/work/cd_expts/tf_obj_det/research
python object_detection/model_main_tf2.py --pipeline_config_path /home/azureuser/work/cd_expts/tf_obj_det/research/object_detection/ts_space/configs/faster_rcnn_resnet101_v1_1024x1024_ts_cotton_tpu-8.config --model_dir /mnt/ts-cvml-datastore/ts_model/models/cotton --checkpoint_dir /mnt/ts-cvml-datastore/ts_model/models/cotton --alsologtostderr --wb_group FRC_NV1_009 --sample_1_of_n_eval_examples 4
