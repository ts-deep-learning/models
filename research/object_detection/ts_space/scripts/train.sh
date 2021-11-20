export PYTHONPATH=/root/work/cd_expts/tf_obj_det_api/research/slim:/root/work/cd_expts/tf_obj_det_api/research/

# ssd inception v3 model
#python /home/ubuntu/work/tf_obj_det/research/object_detection/model_main.py --pipeline_config_path=/home/ubuntu/work/tf_obj_det/research/object_detection/ts_space/configs/ssd_inception_v2_castor.config --model_dir=/home/ubuntu/work/tf_obj_det/research/object_detection/ts_space/trained_models
#python3 object_detection/model_main.py --pipeline_config_path=/home/ubuntu/object_detection_api/train_config/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --model_dir=/home/ubuntu/object_detection_api/detection_benchmark/
#sudo poweroff

# ssd retinanet r101
# python /root/work/cd_expts/tf_obj_det_api/research/object_detection/model_main_tf2.py --pipeline_config_path=/root/work/cd_expts/tf_obj_det_api/research/object_detection/ts_space/configs/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.config --model_dir=/root/work/cd_expts/tf_obj_det_api/research/object_detection/ts_space/trained_models/RTN_E2E_008/

# ssd retinanet r101 evaluation
CUDA_VISIBLE_DEVICES=-1 python /root/work/cd_expts/tf_obj_det_api/research/object_detection/model_main_tf2.py --pipeline_config_path=/root/work/cd_expts/tf_obj_det_api/research/object_detection/ts_space/configs/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.config --model_dir=/root/work/cd_expts/tf_obj_det_api/research/object_detection/ts_space/trained_models/RTN_E2E_008/ --checkpoint_dir=/root/work/cd_expts/tf_obj_det_api/research/object_detection/ts_space/trained_models/RTN_E2E_008/ --sample_1_of_n_eval_examples=1 --alsologtoostderr

