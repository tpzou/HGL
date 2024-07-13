gpuid=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpuid
cd /home/XXXXXXX/A0_TTA-Point/HGL

note="normal"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python adapt_online.py --config_file configs/adaptation/synlidar2kitti_adaptation_model_features.yaml --note $note --use_prototype --use_pseudo_new --pseudo_th 0.7 --pseudo_knn 10 --score_weight --loss_use_score_weight --loss_method_num 1 --loss_eps 0.3 --use_all_pseudo
