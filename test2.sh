rm -rf preds_v2_test
python -m tools.test configs/hrda/cracks_cfg_test.py work_dirs/local-basic/230705_1818_hrda_v2_e28bd/iter_25000.pth --eval mIoU --show-dir preds_v2_test --opacity 0.4