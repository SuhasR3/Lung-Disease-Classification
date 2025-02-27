BATCH_SIZE = 32  
EPOCHS = 50  
LR = 1e-2  
DROPOUT = 0.5  
CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)