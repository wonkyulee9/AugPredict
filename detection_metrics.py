"""
Detection metrics util for evaluation of detection model
"""
import torch

def calc_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA+1) * max(0, yB-yA+1)

    aA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    aB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter / float(aA + aB - inter)
    return iou


def get_mAP(tp, fp, fn):
    prec = tp / (tp + fp)
    prec = torch.where(prec == 0, prec, torch.ones_like(prec))
    rec = tp / (tp + fn)
    prec = prec.permute(1, 0)
    rec = rec.permute(1, 0)
    AP = torch.trapz(prec, rec)
    mAP = AP.mean()
    return mAP


def getnum_tp_fp_fn(targets, predictions):
    tp_count, fp_count, fn_count = torch.zeros([26, 10]), torch.zeros([26, 10]), torch.zeros([26, 10])
    for target, prediction in zip(targets, predictions):
        prediction = sort_by_conf(prediction)
        for c in range(26):
            idx = None
            for i, conf in enumerate(prediction['scores']):
                if conf > c/25:
                    idx = i
                    break
            if idx == None:
                idx = prediction['scores'].shape[0]

            mat = torch.zeros([10, len(target['labels']), len(prediction['labels'][idx:])])
            for i, tar in enumerate(target['boxes']):
                for j, pred in enumerate(prediction['boxes'][idx:]):
                    iou = calc_iou(tar, pred)
                    if iou > 0.5:
                        for k in range(10):
                            if target['labels'][i] == k and k == prediction['labels'][idx:][j]:
                                mat[k][i][j] = 1
            for k in range(10):
                try:
                    tp = mat[k].max(1)[0].sum()
                except:
                    tp = 0
                fp = (prediction['labels'][idx:]==k).sum() - tp
                fn = (target['labels']==k).sum() - tp
                tp_count[c][k] += tp
                fp_count[c][k] += fp
                fn_count[c][k] += fn
    return tp_count, fp_count, fn_count

def sort_by_conf(predictions):
    sort_conf, indices = torch.sort(predictions['scores'])
    predictions['scores'] = sort_conf
    predictions['labels'] = predictions['labels'][indices]
    predictions['boxes'] = predictions['boxes'][indices]
    return predictions
