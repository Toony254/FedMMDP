import torch
import torch.nn.functional as F


def compute_distill_loss(student, teacher, relation_weight=1.0, cosine_weight=0.5, relation_temperature=0.07, loss_scale=16.0):
    student = F.normalize(student.float(), p=2, dim=1)
    teacher = F.normalize(teacher.float(), p=2, dim=1)

    feature_loss = F.mse_loss(student, teacher)
    cosine_loss = 1.0 - F.cosine_similarity(student, teacher, dim=1).mean()

    student_relation = torch.matmul(student, student.t()) / relation_temperature
    teacher_relation = torch.matmul(teacher, teacher.t()) / relation_temperature
    relation_loss = F.kl_div(
        F.log_softmax(student_relation, dim=1),
        F.softmax(teacher_relation, dim=1),
        reduction='batchmean'
    )

    return loss_scale * (feature_loss + cosine_weight * cosine_loss + relation_weight * relation_loss)
