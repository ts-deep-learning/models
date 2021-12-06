export PYTHONPATH=/home/azureuser/work/cd_expts/tf_obj_det/research/slim:/home/azureuser/work/cd_expts/tf_obj_det/research
# Mobilenet 2
# python eval_tf2_model.py -m /mnt/ts-cvml-datastore/ts_model/models/cotton/mobilenet_fpn/CT_MB2F_AZ_002/frozen_model/saved_model -th 0.3 -dr /mnt/ts-cvml-datastore/ts_data/cropped_data/cotton/v1 -j /mnt/ts-cvml-datastore/ts_data/datasets/cotton/dataset_jsons/v1.0.2/test_1.0.1.json
# Mobilenet 1
# python eval_tf2_model.py -m /mnt/ts-cvml-datastore/ts_model/models/cotton/mobilenet_fpn/CT_MBF_AZ_002/frozen_model/saved_model -th 0.3 -dr /mnt/ts-cvml-datastore/ts_data/cropped_data/cotton/v1 -j /mnt/ts-cvml-datastore/ts_data/datasets/cotton/dataset_jsons/v1.0.2/test_1.0.1.json
# Retinanet
# python eval_tf2_model.py -m /mnt/ts-cvml-datastore/ts_model/models/cotton/RTN_E2E_008/frozen_model/saved_model -th 0.3 -dr /mnt/ts-cvml-datastore/ts_data/cropped_data/cotton/v1 -j /mnt/ts-cvml-datastore/ts_data/datasets/cotton/dataset_jsons/v1.0.2/dev_1.0.1.json
# FRCNN
python eval_tf2_model.py -m /mnt/ts-cvml-datastore/ts_model/models/cotton/FRC_NV1_009/frozen_model/saved_model -th 0.3 -dr /mnt/ts-cvml-datastore/ts_data/cropped_data/cotton/v1 -j /mnt/ts-cvml-datastore/ts_data/datasets/cotton/dataset_jsons/v1.0.2/test_1.0.1.json
