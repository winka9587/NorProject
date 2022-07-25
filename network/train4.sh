echo ensure you are in path
echo .../NorProject/network/
echo now:
pwd
python train_ae.py \
            --CDLossTimes=1.0 \
            --CorrLossTimes=1.0 \
            --EntropyLossTimes=10.0 \
            --device_ids="4"