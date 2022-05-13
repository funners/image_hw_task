# -*- coding = utf-8 -*-
# @File_name = d2l
import numpy as np
import matplotlib.pyplot as plt
historys_update = np.load("history_u_net_update.npy", allow_pickle=True)
historys = np.load("history_u_net.npy", allow_pickle=True)

u_net_loss = []
u_net_val_loss = []
u_net_iou_coef = []
u_net_val_iou_coef = []
u_net_acc = []
u_net_val_acc = []
u_net_dice_coef = []
u_net_val_dice_coef = []

u_net_update_loss = []
u_net_update_val_loss = []
u_net_update_iou_coef = []
u_net_update_val_iou_coef = []
u_net_update_acc = []
u_net_update_val_acc = []
u_net_update_dice_coef = []
u_net_update_val_dice_coef = []

for i in range(len(historys)):
    u_net_loss.append(historys[i]["loss"][0])
    u_net_val_loss.append(historys[i]['val_loss'][0])
    u_net_iou_coef.append(historys[i]['iou_coef'][0])
    u_net_val_iou_coef.append(historys[i]['val_iou_coef'][0])
    u_net_acc.append(historys[i]['acc'][0])
    u_net_val_acc.append(historys[i]['val_acc'][0])
    u_net_dice_coef.append(historys[i]['dice_coef'][0])
    u_net_val_dice_coef.append(historys[i]['val_dice_coef'][0])


for i in range(len(historys_update)):
    u_net_update_loss.append(historys_update[i]["loss"][0])
    u_net_update_val_loss.append(historys_update[i]["val_loss"][0])
    u_net_update_iou_coef.append(historys_update[i]["iou_coef"][0])
    u_net_update_val_iou_coef.append(historys_update[i]["val_iou_coef"][0])
    u_net_update_acc.append(historys_update[i]["acc"][0])
    u_net_update_val_acc.append(historys_update[i]["val_acc"][0])
    u_net_update_dice_coef.append(historys_update[i]["dice_coef"][0])
    u_net_update_val_dice_coef.append(historys_update[i]["val_dice_coef"][0])



plt.figure()
plt.plot(u_net_loss, label='loss')
plt.plot(u_net_val_loss, label='val_loss')
plt.plot(u_net_update_loss, label="update loss")
plt.plot(u_net_update_val_loss, label="update loss")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid', "update train", "update valid"], loc='upper left')
plt.savefig('./plot/loss.png')
plt.close()

plt.figure()
plt.plot(u_net_iou_coef, label='iou')
plt.plot(u_net_val_iou_coef, label='val_iou')
plt.plot(u_net_update_iou_coef, label='update iou')
plt.plot(u_net_update_val_iou_coef, label='update val_iou')
plt.title('model iou coef')
plt.ylabel('iou coef')
plt.xlabel('epoch')
plt.legend(['train', 'valid', "update train", "update valid"], loc='upper left')
plt.savefig('./plot/iou_coef.png')
plt.close()


plt.figure()
plt.plot(u_net_acc, label='acc')
plt.plot(u_net_val_acc, label='val_acc')
plt.plot(u_net_update_acc, label='update acc')
plt.plot(u_net_update_val_acc, label='update val_acc')
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'valid', "update train", "update valid"], loc='upper left')
plt.savefig('./plot/acc.png')
plt.close()

plt.figure()
plt.plot(u_net_dice_coef, label='dice')
plt.plot(u_net_val_dice_coef, label='val_dice')
plt.plot(u_net_update_dice_coef, label='update dice')
plt.plot(u_net_update_val_dice_coef, label='update val_dice')
plt.title('model dice')
plt.ylabel('dice')
plt.xlabel('epoch')
plt.legend(['train', 'valid', "update train", "update valid"], loc='upper left')
plt.savefig('./plot/dice.png')
plt.close()
print("hello world")