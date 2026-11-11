Test set: 1593 exams from 914 patients, 1108 with at least one fracture box.

| Subgroup | Exams | AUC |
|---|---|---|
| all | 1593 | 0.990 (0.985-0.994) |
| initial_exam | 876 | 0.984 (0.976-0.991) |
| follow_up | 717 | 0.976 (0.946-0.993) |
| certain_labels | 1554 | 0.993 (0.989-0.996) |
| two_views | 1396 | 0.990 (0.985-0.994) |

Image level: AUC 0.989 (0.984-0.992) on 3015 images, 2017 with a fracture box.

| Operating point (val) | Threshold | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| spec_0.80 | 0.025 | 0.989 (0.983-0.995) | 0.765 (0.720-0.809) | 0.906 (0.884-0.926) | 0.969 (0.950-0.984) |
| spec_0.90 | 0.142 | 0.979 (0.970-0.987) | 0.895 (0.861-0.927) | 0.955 (0.940-0.970) | 0.950 (0.928-0.968) |
| spec_0.95 | 0.488 | 0.958 (0.945-0.970) | 0.951 (0.929-0.971) | 0.978 (0.968-0.987) | 0.909 (0.881-0.935) |

| Specificity on the test ROC | Sensitivity |
|---|---|
| 0.80 | 0.989 |
| 0.90 | 0.977 |
| 0.95 | 0.961 |

Lesion level (2683 fracture boxes, IoU >= 0.3):

| FP per image | Sensitivity |
|---|---|
| 0.25 | 0.958 |
| 0.5 | 0.975 |
| 1.0 | 0.984 |
| 2.0 | 0.988 |

mAP50 (all classes) 0.513, mAP50-95 0.318, fracture AP50 0.936
