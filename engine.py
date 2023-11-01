"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if isinstance(outputs, list):
                loss_list = [criterion(o, targets) / len(outputs) for o in outputs]
                loss = sum(loss_list)
            else:
                loss = criterion(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if isinstance(outputs, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device,tt):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # Conformer
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        if isinstance(output, list):
            # Conformer
            acc1_head1 = accuracy(output[0], target, topk=(1,))[0]
            acc1_head2 = accuracy(output[1], target, topk=(1,))[0]
            acc1_total = accuracy(output[0] + output[1], target, topk=(1,))[0]
            predicted_labels = np.argmax((output[0] + output[1]).cpu().numpy(),axis=1)

        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        if tt == True:
            output_file = "metrics.txt"
            # Create the confusion matrix
            output1 = (predicted_labels)
            target = target.cpu().numpy()
            conf_matrix = confusion_matrix(target, output1)
            classs = ["Acne", "Bullous Disease", "Eczema", "Utricaria Hives"]

            # Calculate metrics
            accuracy1 = accuracy_score(target, output1)
            precision = precision_score(target, output1, average='macro')
            recall = recall_score(target, output1, average='macro')
            f1 = f1_score(target, output1, average='macro')

            # Calculate class-specific metrics
            class_accuracies = []
            class_precisions = []
            class_recalls = []
            class_f1_scores = []

            # Add code to calculate ROC-AUC for each class
            class_roc_auc = []

            for i in range(conf_matrix.shape[0]):
                class_accuracy = conf_matrix[i, i] / conf_matrix[i, :].sum()
                class_accuracies.append(class_accuracy)

                class_precision = precision_score(target, output1, labels=[i], average=None)
                class_precisions.append(class_precision)

                class_recall = recall_score(target, output1, labels=[i], average=None)
                class_recalls.append(class_recall)

                class_f1 = f1_score(target, output1, labels=[i], average=None)
                class_f1_scores.append(class_f1)

                # Calculate ROC curve and AUC for the current class
                fpr, tpr, _ = roc_curve(target, output1, pos_label=i)
                roc_auc = auc(fpr, tpr)
                class_roc_auc.append(roc_auc)

            with open(output_file, "w") as file:
                file.write("Class-specific metrics:\n")
                for class_index, (accuracy2, precision2, recall2, f1_score1, roc_auc) in enumerate(
                        zip(class_accuracies, class_precisions, class_recalls, class_f1_scores, class_roc_auc)):
                    class_name = classs[class_index]
                    file.write(f"{class_name} - Accuracy: {accuracy2}\n")
                    file.write(f"{class_name} - Precision: {precision2[0]}\n")
                    file.write(f"{class_name} - Recall: {recall2[0]}\n")
                    file.write(f"{class_name} - F1-Score: {f1_score1[0]}\n")
                    file.write(f"{class_name} - ROC-AUC: {roc_auc}\n")

                file.write("Confusion Matrix:\n")
                for row in conf_matrix:
                    file.write(" ".join(map(str, row)) + "\n")

                file.write("Accuracy: {}\n".format(accuracy1))
                file.write("Precision: {}\n".format(precision))
                file.write("Recall: {}\n".format(recall))
                file.write("F1-Score: {}\n".format(f1))

            # Plot ROC-AUC curves and save them as images
            for class_index, (fpr, tpr, roc_auc) in enumerate(zip(fpr, tpr, class_roc_auc)):
                class_name = classs[class_index]
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {class_name}')
                plt.legend(loc="lower right")
                plt.savefig(f"{class_name}_roc_curve.png")
                plt.close()

        # print(f"Metrics saved to {output_file}")



        else:
            # others
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        if isinstance(output, list):
            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
            metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
            metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)
        else:
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              .format(heads_top1=metric_logger.acc1, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
