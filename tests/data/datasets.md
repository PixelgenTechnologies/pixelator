# Datasets


[//]: # ( TODO: replace this with a public dataset)
[//]: # ( TODO: remove private fields from panel)

## uropod_control

This dataset is a subsample of 300k reads from PGSeq98 (230420_VH00725_91_AACLYY2M5).
Data was subsampled with seqkit (using the default random seed: 11)

The matching panel is `UNO_D21_conjV21.csv`

```bash
seqkit sample Uropod_control_S1_R1_001.fastq.gz -2 -n 300000 -o uropod_control_300k_S1_R1_001.fastq.gz
seqkit sample Uropod_control_S1_R2_001.fastq.gz -2 -n 300000 -o uropod_control_300k_S1_R2_001.fastq.gz
```
