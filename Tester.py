import torch
import numpy as np
import os
import time
import torch.nn as nn
from tqdm import tqdm
from utils.util import AverageMeter, ensure_dir, object_based_infer
import shutil
from utils.metrics import Evaluator_tensor

class Tester(object):
    def __init__(self,
                 model,
                 config,
                 args,
                 test_data_loader,
                 class_name,
                 begin_time,
                 resume_file
                 ):

        # for general
        self.config = config
        self.args = args
        self.device = torch.device('cpu') if self.args.gpu == -1 else torch.device('cuda:{}'.format(self.args.gpu))
        self.class_name = class_name
        # for Test
        if isinstance(model, list):
            self.model = []
            for m in model:
                m=m.to(self.device)
                self.model.append(m)
        else:
            self.model = model.to(self.device)

        self.models = []

        # for time
        self.begin_time = begin_time

        # for data
        self.test_data_loader = test_data_loader

        # for resume/save path
        self.history = {
            "eval": {
                "loss": [],
                "acc": [],
                "miou": [],
                "time": [],
                "prec": [],
                "recall": [],
                "f_score": [],
            },
        }

        self.model_name = self.config.model_name

        if self.config.use_seed:
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time + '_seed' + str(self.config.random_seed), 'log')
        else:
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time, 'log')

        if not self.args.only_prediction:
            self.test_log_path = os.path.join(self.args.output, 'test', 'log', self.model_name,
                                              self.begin_time)
            ensure_dir(self.test_log_path)

        self.predict_path = os.path.join(self.args.output, 'test', 'predict', self.model_name,
                                         self.begin_time)
        ensure_dir(self.predict_path)

        if self.config.use_seed:
            self.resume_ckpt_path = resume_file if resume_file is not None else \
                os.path.join(self.config.save_dir, self.model_name,
                             self.begin_time + '_seed' + str(self.config.random_seed), 'checkpoint-best.pth')
        else:
            self.resume_ckpt_path = resume_file if resume_file is not None else \
                os.path.join(self.config.save_dir, self.model_name,
                             self.begin_time, 'checkpoint-best.pth')

        with open(os.path.join(self.predict_path, 'checkpoint.txt'), 'w') as f:
            f.write(self.resume_ckpt_path)
            self.model_name

        self.evaluator = Evaluator_tensor(self.config.nb_classes, self.device)
        self.evaluator_BD = Evaluator_tensor(2, self.device)

    def eval_and_predict_damage_PDMT(self):
        """
        pixel-based evaluation
        :return:
        """
        self._resume_ckpt_PSMT()
        self.evaluator.reset()
        self.evaluator_BD.reset()

        ave_total_loss = AverageMeter()
        bce_loss = nn.BCEWithLogitsLoss()
        ce_loss = nn.CrossEntropyLoss()

        self.model[0].eval()
        self.model[1].eval()

        with torch.no_grad():
            tic = time.time()
            for steps, (data, target, filenames) in tqdm(enumerate(self.test_data_loader, start=1)):
                # data
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                logits0= self.model[0](data)
                logits1= self.model[1](data)
                logits = {}
                logits['damage'] = (logits0['damage'] + logits1['damage']) / 2
                logits['building'] = (logits0['building'] + logits1['building']) / 2

                target_BD = target.clone().detach()
                target_BD[target_BD != 0] = 1

                logits_BD=torch.squeeze(logits['building'], dim=1)
                probability = torch.sigmoid(logits_BD)
                loss_BD = bce_loss(logits_BD, target_BD)

                target = target.long()
                loss_D = ce_loss(logits['damage'], target)
                loss = loss_BD + loss_D

                # building
                binary_map = probability.clone().detach()
                binary_map[binary_map < 0.5] = 0
                binary_map[binary_map >= 0.5] = 1
                pred_BD = binary_map.contiguous().view(-1).long()

                # building damage
                pred_D = torch.argmax(logits['damage'], dim=1)
                pred_D = pred_D.view(-1).long()

                label_BD = target_BD.view(-1).long()
                label_D = target.view(-1).long()

                # Add batch sample into evaluator
                self.evaluator.add_batch(label_D, pred_D)
                self.evaluator_BD.add_batch(label_BD, pred_BD)

                ave_total_loss.update(loss.item())

            total_time = time.time() - tic

            # Building
            acc0 = self.evaluator_BD.Pixel_Accuracy().cpu().detach().numpy()
            acc_class0 = self.evaluator_BD.Pixel_Accuracy_Class().cpu().detach().numpy()
            miou0 = self.evaluator_BD.Mean_Intersection_over_Union().cpu().detach().numpy()
            fwiou0 = self.evaluator_BD.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
            confusion_matrix0 = self.evaluator_BD.get_confusion_matrix().cpu().detach().numpy()
            TP0, FP0, FN0, TN0 = self.evaluator_BD.get_base_value()
            iou0 = self.evaluator_BD.get_iou().cpu().detach().numpy()
            prec0 = self.evaluator_BD.Pixel_Precision_Class().cpu().detach().numpy()
            recall0 = self.evaluator_BD.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score0 = self.evaluator_BD.Pixel_F1_score_Class().cpu().detach().numpy()
            kappa_coe0 = self.evaluator_BD.Kapaa_coefficient().cpu().detach().numpy()

            # building damage
            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            acc_class = self.evaluator.Pixel_Accuracy_Class().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
            confusion_matrix1 = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
            TP, FP, FN, TN = self.evaluator.get_base_value()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()
            kappa_coe = self.evaluator.Kapaa_coefficient().cpu().detach().numpy()

            # display evaluation result
            print('Evaluation phase !\n'
                  'Accuracy: {:6.4f}, Loss: {:.6f}'.format(
                acc, ave_total_loss.average()))
            np.set_printoptions(formatter={'int': '{: 9}'.format})
            print('Class:    ', self.class_name, ' Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            print('IoU:      ', np.hstack((iou, np.average(iou))))
            print('Precision:', np.hstack((prec, np.average(prec))))
            print('Recall:   ', np.hstack((recall, np.average(recall))))
            print('F_Score:  ', np.hstack((f1_score, np.average(f1_score))))
            np.set_printoptions(formatter={'int': '{:14}'.format})
            print('Confusion_matrix:')
            print(confusion_matrix1)
            # normalized confusion matrix
            np.set_printoptions(formatter={'float': '{: 7.4f}'.format})
            confusion_matrix_norm = confusion_matrix1 / np.sum(confusion_matrix1)
            print('Normalized_confusion_matrix:')
            print(confusion_matrix_norm)
            print('Kappa_Coefficient:{:10.6f}'.format(kappa_coe))

            # building
            print('\n')
            print('Accuracy: {:6.4f}'.format(acc0))
            np.set_printoptions(formatter={'int': '{: 9}'.format})
            print('Class:    Background,  Building,   Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            print('IoU:      ', np.hstack((iou0, np.average(iou0))))
            print('Precision:', np.hstack((prec0, np.average(prec0))))
            print('Recall:   ', np.hstack((recall0, np.average(recall0))))
            print('F_Score:  ', np.hstack((f1_score0, np.average(f1_score0))))
            print('FW_IoU:   ', fwiou0)
            np.set_printoptions(formatter={'int': '{:14}'.format})
            print('Confusion_matrix:')
            print(confusion_matrix0)
            # normalized confusion matrix
            np.set_printoptions(formatter={'float': '{: 7.4f}'.format})
            confusion_matrix_norm0 = confusion_matrix0 / np.sum(confusion_matrix0)
            print('Normalized_confusion_matrix:')
            print(confusion_matrix_norm0)
            print('Kappa_Coefficient:{:10.6f}'.format(kappa_coe0))


            print('Prediction Phase !\n'
                  'Total Time cost: {:.2f}s\n'
                  .format(total_time,
                          ))
        self.history["eval"]["loss"].append(ave_total_loss.average())
        self.history["eval"]["acc"].append(acc.tolist())
        self.history["eval"]["miou"].append(iou.tolist())
        self.history["eval"]["time"].append(total_time)

        self.history["eval"]["prec"].append(prec.tolist())
        self.history["eval"]["recall"].append(recall.tolist())
        self.history["eval"]["f_score"].append(f1_score.tolist())

        print("     + Saved history of evaluation phase !")
        hist_path = os.path.join(self.test_log_path, "history1.txt")
        with open(hist_path, 'w') as f:
            f.write(str(self.history).replace("'", '"'))
            # building damage
            f.write('\n***************************************************')
            f.write('\n[Building damage seg accuracy]')
            f.write('\nAccuracy:{:10.6f}'.format(acc))
            f.write('\nKappa_Coefficient:{:10.6f}'.format(kappa_coe))
            f.write('\nConfusion_matrix:\n')
            f.write(str(confusion_matrix1))
            np.set_printoptions(formatter={'float': '{: 6.3f}'.format})
            f.write('\n Normalized_confusion_matrix:\n')
            f.write(str(confusion_matrix_norm))

            np.set_printoptions(formatter={'int': '{: 9}'.format})
            num = np.arange(0, self.config.nb_classes)
            f.write('\nClass:    ' + str(self.class_name) + '  Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            format_iou = np.hstack((iou, np.average(iou)))
            format_prec = np.hstack((prec, np.average(prec)))
            format_recall = np.hstack((recall, np.average(recall)))
            format_f1_score = np.hstack((f1_score, np.average(f1_score)))
            f.write('\nIoU:      ' + str(format_iou))
            f.write('\nPrecision:' + str(format_prec))
            f.write('\nRecall:   ' + str(format_recall))
            f.write('\nF1_score: ' + str(format_f1_score))
            f.write('\nFW_IoU:   ' + str(fwiou))

            # building
            f.write('\n***************************************************')
            f.write('\n[Building seg accuracy]')
            f.write('\nAccuracy:{:10.6f}'.format(acc0))
            f.write('\nKappa_Coefficient:{:10.6f}'.format(kappa_coe0))
            f.write('\nConfusion_matrix:\n')
            f.write(str(confusion_matrix0))
            np.set_printoptions(formatter={'float': '{: 6.3f}'.format})
            f.write('\n Normalized_confusion_matrix:\n')
            f.write(str(confusion_matrix_norm0))

            np.set_printoptions(formatter={'int': '{: 9}'.format})
            num = np.arange(0, 2)
            f.write('\nClass:    Background,  Building,   Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            format_iou0 = np.hstack((iou0, np.average(iou0)))
            format_prec0 = np.hstack((prec0, np.average(prec0)))
            format_recall0 = np.hstack((recall0, np.average(recall0)))
            format_f1_score0 = np.hstack((f1_score0, np.average(f1_score0)))
            f.write('\nIoU:      ' + str(format_iou0))
            f.write('\nPrecision:' + str(format_prec0))
            f.write('\nRecall:   ' + str(format_recall0))
            f.write('\nF1_score: ' + str(format_f1_score0))
            f.write('\nFW_IoU:   ' + str(fwiou0))

        test_log_path1 = os.path.join(self.args.output, 'test', 'log', self.model_name, "history1.txt")
        if os.path.exists(test_log_path1):
            os.remove(test_log_path1)
        shutil.copy(hist_path, test_log_path1)
        if not self.args.is_test:
            hist_test_log_path = os.path.join(self.log_dir, "history1-test.txt")
            shutil.copy(hist_path, hist_test_log_path)
        else:
            input_dir_path = os.path.dirname(self.resume_ckpt_path)
            input_file_name = os.path.basename(self.resume_ckpt_path)
            output_dir = os.path.join(input_dir_path, 'batch_test')
            ensure_dir(output_dir)
            output_file_path = os.path.join(output_dir, input_file_name + '.txt')
            shutil.copy(hist_path, output_file_path)
        output_iou = {}
        output_iou['miou_damage'] = miou
        output_iou['iou_building'] = iou0[1]
        return output_iou

    def eval_and_predict_damage_PDMT_object(self):
        """
        object-based evaluation
        :return:
        """
        self._resume_ckpt_PSMT()
        self.evaluator.reset()
        self.evaluator_BD.reset()

        ave_total_loss = AverageMeter()
        bce_loss = nn.BCEWithLogitsLoss()
        ce_loss=nn.CrossEntropyLoss()

        self.model[0].eval()
        self.model[1].eval()

        with torch.no_grad():
            tic = time.time()
            for steps, (data, target, filenames) in tqdm(enumerate(self.test_data_loader, start=1)):
                # data
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                logits0= self.model[0](data)
                logits1= self.model[1](data)
                logits = {}
                logits['damage'] = (logits0['damage'] + logits1['damage']) / 2
                logits['building'] = (logits0['building'] + logits1['building']) / 2

                # vote by object
                mask_BD, mask_D = object_based_infer(logits['building'], logits['damage'])
                mask_D = torch.from_numpy(mask_D).to(self.device, non_blocking=True)

                target_BD = target.clone().detach()
                target_BD[target_BD != 0] = 1

                # building seg
                logits_BD=torch.squeeze(logits['building'], dim=1)
                probability = torch.sigmoid(logits_BD)
                loss_BD = bce_loss(logits_BD, target_BD)

                # damage seg
                target = target.long()
                loss_D = ce_loss(logits['damage'], target)

                loss = loss_BD + loss_D

                # 建筑物
                binary_map = probability.clone().detach()
                binary_map[binary_map < 0.5] = 0
                binary_map[binary_map >= 0.5] = 1
                pred_BD = binary_map.contiguous().view(-1).long()

                pred_D = torch.argmax(logits['damage'], dim=1)
                pred_D = mask_D
                pred_D = pred_D.view(-1).long()

                label_BD = target_BD.view(-1).long()
                label_D = target.view(-1).long()

                # Add batch sample into evaluator
                self.evaluator.add_batch(label_D, pred_D)
                self.evaluator_BD.add_batch(label_BD, pred_BD)

                ave_total_loss.update(loss.item())

            total_time = time.time() - tic

            # Building acc
            acc0 = self.evaluator_BD.Pixel_Accuracy().cpu().detach().numpy()
            acc_class0 = self.evaluator_BD.Pixel_Accuracy_Class().cpu().detach().numpy()
            miou0 = self.evaluator_BD.Mean_Intersection_over_Union().cpu().detach().numpy()
            fwiou0 = self.evaluator_BD.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
            confusion_matrix0 = self.evaluator_BD.get_confusion_matrix().cpu().detach().numpy()
            TP0, FP0, FN0, TN0 = self.evaluator_BD.get_base_value()
            iou0 = self.evaluator_BD.get_iou().cpu().detach().numpy()
            prec0 = self.evaluator_BD.Pixel_Precision_Class().cpu().detach().numpy()
            recall0 = self.evaluator_BD.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score0 = self.evaluator_BD.Pixel_F1_score_Class().cpu().detach().numpy()
            kappa_coe0 = self.evaluator_BD.Kapaa_coefficient().cpu().detach().numpy()

            # building damage acc
            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            acc_class = self.evaluator.Pixel_Accuracy_Class().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
            confusion_matrix1 = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
            TP, FP, FN, TN = self.evaluator.get_base_value()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()
            kappa_coe = self.evaluator.Kapaa_coefficient().cpu().detach().numpy()


            # display evaluation result
            print('Evaluation phase !\n'
                  'Accuracy: {:6.4f}, Loss: {:.6f}'.format(
                acc, ave_total_loss.average()))
            np.set_printoptions(formatter={'int': '{: 9}'.format})
            print('Class:    ', self.class_name, ' Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            print('IoU:      ', np.hstack((iou, np.average(iou))))
            print('Precision:', np.hstack((prec, np.average(prec))))
            print('Recall:   ', np.hstack((recall, np.average(recall))))
            print('F_Score:  ', np.hstack((f1_score, np.average(f1_score))))
            np.set_printoptions(formatter={'int': '{:14}'.format})
            print('Confusion_matrix:')
            print(confusion_matrix1)
            # normalized confusion matrix
            np.set_printoptions(formatter={'float': '{: 7.4f}'.format})
            confusion_matrix_norm = confusion_matrix1 / np.sum(confusion_matrix1)
            print('Normalized_confusion_matrix:')
            print(confusion_matrix_norm)
            print('Kappa_Coefficient:{:10.6f}'.format(kappa_coe))

            # building
            print('\n')
            print('Accuracy: {:6.4f}'.format(acc0))
            np.set_printoptions(formatter={'int': '{: 9}'.format})
            print('Class:    Background,  Building,   Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            print('IoU:      ', np.hstack((iou0, np.average(iou0))))
            print('Precision:', np.hstack((prec0, np.average(prec0))))
            print('Recall:   ', np.hstack((recall0, np.average(recall0))))
            print('F_Score:  ', np.hstack((f1_score0, np.average(f1_score0))))
            print('FW_IoU:   ', fwiou0)
            np.set_printoptions(formatter={'int': '{:14}'.format})
            print('Confusion_matrix:')
            print(confusion_matrix0)
            # normalized confusion matrix
            np.set_printoptions(formatter={'float': '{: 7.4f}'.format})
            confusion_matrix_norm0 = confusion_matrix0 / np.sum(confusion_matrix0)
            print('Normalized_confusion_matrix:')
            print(confusion_matrix_norm0)
            print('Kappa_Coefficient:{:10.6f}'.format(kappa_coe0))

            print('Prediction Phase !\n'
                  'Total Time cost: {:.2f}s\n'
                  .format(total_time,
                          ))
        self.history["eval"]["loss"].append(ave_total_loss.average())
        self.history["eval"]["acc"].append(acc.tolist())
        self.history["eval"]["miou"].append(iou.tolist())
        self.history["eval"]["time"].append(total_time)

        self.history["eval"]["prec"].append(prec.tolist())
        self.history["eval"]["recall"].append(recall.tolist())
        self.history["eval"]["f_score"].append(f1_score.tolist())

        # save results to log file
        print("     + Saved history of evaluation phase !")
        hist_path = os.path.join(self.test_log_path, "history1.txt")
        with open(hist_path, 'w') as f:
            f.write(str(self.history).replace("'", '"'))
            # building damage
            f.write('\n***************************************************')
            f.write('\n[Building damage seg accuracy]')
            f.write('\nAccuracy:{:10.6f}'.format(acc))
            f.write('\nKappa_Coefficient:{:10.6f}'.format(kappa_coe))
            f.write('\nConfusion_matrix:\n')
            f.write(str(confusion_matrix1))
            np.set_printoptions(formatter={'float': '{: 6.3f}'.format})
            f.write('\n Normalized_confusion_matrix:\n')
            f.write(str(confusion_matrix_norm))

            np.set_printoptions(formatter={'int': '{: 9}'.format})
            num = np.arange(0, self.config.nb_classes)
            f.write('\nClass:    ' + str(self.class_name) + '  Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            format_iou = np.hstack((iou, np.average(iou)))
            format_prec = np.hstack((prec, np.average(prec)))
            format_recall = np.hstack((recall, np.average(recall)))
            format_f1_score = np.hstack((f1_score, np.average(f1_score)))
            f.write('\nIoU:      ' + str(format_iou))
            f.write('\nPrecision:' + str(format_prec))
            f.write('\nRecall:   ' + str(format_recall))
            f.write('\nF1_score: ' + str(format_f1_score))
            f.write('\nFW_IoU:   ' + str(fwiou))

            # building
            f.write('\n***************************************************')
            f.write('\n[Building seg accuracy]')
            f.write('\nAccuracy:{:10.6f}'.format(acc0))
            f.write('\nKappa_Coefficient:{:10.6f}'.format(kappa_coe0))
            f.write('\nConfusion_matrix:\n')
            f.write(str(confusion_matrix0))
            np.set_printoptions(formatter={'float': '{: 6.3f}'.format})
            f.write('\n Normalized_confusion_matrix:\n')
            f.write(str(confusion_matrix_norm0))

            np.set_printoptions(formatter={'int': '{: 9}'.format})
            num = np.arange(0, 2)
            f.write('\nClass:    Background,  Building,   Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            format_iou0 = np.hstack((iou0, np.average(iou0)))
            format_prec0 = np.hstack((prec0, np.average(prec0)))
            format_recall0 = np.hstack((recall0, np.average(recall0)))
            format_f1_score0 = np.hstack((f1_score0, np.average(f1_score0)))
            f.write('\nIoU:      ' + str(format_iou0))
            f.write('\nPrecision:' + str(format_prec0))
            f.write('\nRecall:   ' + str(format_recall0))
            f.write('\nF1_score: ' + str(format_f1_score0))
            f.write('\nFW_IoU:   ' + str(fwiou0))

        test_log_path1 = os.path.join(self.args.output, 'test', 'log', self.model_name, "history1.txt")
        # 直接复制
        if os.path.exists(test_log_path1):
            os.remove(test_log_path1)
        shutil.copy(hist_path, test_log_path1)
        if not self.args.is_test:
            hist_test_log_path = os.path.join(self.log_dir, "history1-test.txt")
            shutil.copy(hist_path, hist_test_log_path)
        else:
            input_dir_path = os.path.dirname(self.resume_ckpt_path)
            input_file_name = os.path.basename(self.resume_ckpt_path)
            output_dir = os.path.join(input_dir_path, 'batch_test')
            ensure_dir(output_dir)
            output_file_path = os.path.join(output_dir, input_file_name + '.txt')
            shutil.copy(hist_path, output_file_path)
        output_iou = {}
        output_iou['miou_damage'] = miou
        output_iou['iou_building'] = iou0[1]
        return output_iou

    def _resume_ckpt_PSMT(self):
        print("     + Loading ckpt path : {} ...".format(self.resume_ckpt_path))
        checkpoint = torch.load(self.resume_ckpt_path)
        self.model[0].load_state_dict(checkpoint['state_dict1'], strict=True)
        self.model[1].load_state_dict(checkpoint['state_dict2'], strict=True)
        print("     + Model State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Loaded ! \n"
              "     + Prepare to test ! ! !"
              .format(self.resume_ckpt_path))

