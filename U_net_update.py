# -*- coding = utf-8 -*-
# @File_name = U_net
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from image_preprocessing import highPassFilter
from model import U_net_with_hf
from metrics import iou_coef, dice_coef
from utils import predict
batch_size = 8
IMAGE_SIZE = 256
epochs = 16

# 读取数据路径
np.random.seed(10000)
covid_image_path = './COVID-19_Radiography_Dataset/COVID/images/'
normal_image_path = './COVID-19_Radiography_Dataset/Normal/images/'
pneumonia_image_path = './COVID-19_Radiography_Dataset/Viral Pneumonia/images/'
lung_opacity_image_path = './COVID-19_Radiography_Dataset/Lung_Opacity/images/'
covid_mask_path = './COVID-19_Radiography_Dataset/COVID/masks/'
normal_mask_path = './COVID-19_Radiography_Dataset/Normal/masks/'
pneumonia_mask_path = './COVID-19_Radiography_Dataset/Viral Pneumonia/masks/'
lung_opacity_mask_path = './COVID-19_Radiography_Dataset/Lung_Opacity/masks/'
all_image_paths = [[covid_image_path+file for file in os.listdir(covid_image_path)]
                   +[normal_image_path+file for file in os.listdir(normal_image_path)]
                   +[pneumonia_image_path+file for file in os.listdir(pneumonia_image_path)]
                   +[lung_opacity_image_path+file for file in os.listdir(lung_opacity_image_path)]
                  ][0]
all_mask_paths = [[covid_mask_path+file for file in os.listdir(covid_mask_path)]
                  +[normal_mask_path+file for file in os.listdir(normal_mask_path)]
                  +[pneumonia_mask_path+file for file in os.listdir(pneumonia_mask_path)]
                  +[lung_opacity_mask_path+file for file in os.listdir(lung_opacity_mask_path)]
                 ][0]
predict_show_image_path = all_image_paths[: 100]
predict_show_mask_path = all_mask_paths[: 100]

# 打乱数据集顺序
all_image_paths, all_mask_paths = shuffle(all_image_paths, all_mask_paths)


def open_images(paths, high_freq=False):
    images = []
    if high_freq:
        for path in paths:
            image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode="grayscale")
            image = np.array(image)
            high_image = highPassFilter(image, 16)
            image = image.reshape([IMAGE_SIZE, IMAGE_SIZE, 1])
            high_image = high_image.reshape([IMAGE_SIZE, IMAGE_SIZE, 1])
            image = np.concatenate([image, high_image], axis=2)
            images.append(image)
    else:
        for path in paths:
            image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            image = np.mean(image, axis=-1) / 255.0
            images.append(image)
    return np.array(images)


print('Total Number of Samples:', len(all_image_paths))

def datagen(image_paths, mask_paths, batch_size=16):
    for x in range(0, len(image_paths), batch_size):
        images = open_images(image_paths[x:x+batch_size], high_freq=True).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 2)
        masks = open_images(mask_paths[x:x+batch_size]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        yield images, masks

train_image_paths = all_image_paths[:1000]
train_mask_paths = all_mask_paths[:1000]
val_image_paths = all_image_paths[1000:1500]
val_mask_paths = all_mask_paths[1000:1500]


model = U_net_with_hf()
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[iou_coef, "acc", dice_coef])

filepath = "best_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_iou_coef', save_weights_only=True, verbose=1, save_best_only=True)
if os.path.exists(filepath):
    model.load_weights(filepath)
    print("checkpoint_loaded")


historys = []
for _ in range(epochs):
    history = model.fit(datagen(train_image_paths, train_mask_paths, batch_size=batch_size), epochs=1,
                        steps_per_epoch=int(len(train_image_paths)//batch_size),
                        validation_data=datagen(val_image_paths, val_mask_paths, batch_size=batch_size),
                        shuffle=True, callbacks=[checkpoint])
    historys.append(history.history)
historys = np.array(historys)
np.save('history_u_net_update.npy', historys)
model.evaluate(datagen(val_image_paths, val_mask_paths, batch_size=batch_size),
               steps=int(len(val_image_paths)//batch_size), batch_size=batch_size)
model.save("saved_u_net_update_model/")


# 输出预测分割结果图像
c = 2
r = 5
fig = plt.figure(figsize=(8, r * 4))
for i in range(1, c * r + 1, 2):
    image = open_images([predict_show_image_path[i-1]]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    imageinput = open_images([predict_show_image_path[i-1]], high_freq=True).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 2)
    mask = open_images([predict_show_mask_path[i - 1]]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    pred = predict(model, imageinput)
    fig.add_subplot(r, c, i)
    plt.axis('off')
    plt.title('Actual')
    plt.imshow(image[0], cmap='gray', interpolation='none')
    plt.imshow(mask[0], cmap='Spectral_r', alpha=0.3)

    fig.add_subplot(r, c, i + 1)
    plt.axis('off')
    plt.title('Predicted')
    plt.imshow(image[0], cmap='gray', interpolation='none')
    plt.imshow(pred[0], cmap='Spectral_r', alpha=0.3)
plt.savefig("./u_net_update_plot/predict_update.png")


loss = []
val_loss = []
iou_coef = []
val_iou_coef = []
acc = []
val_acc = []
for i in range(len(historys)):
    loss.append(historys[i]["loss"])
    val_loss.append(historys[i]['val_loss'])
    iou_coef.append(historys[i]['iou_coef'])
    val_iou_coef.append(historys[i]['val_iou_coef'])
    acc.append(historys[i]['acc'])
    val_acc.append(historys[i]['val_acc'])

# 输出loss/iou/acc曲线
plt.figure()
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./u_net_update_plot/loss.png')
plt.close()

plt.figure()
plt.plot(iou_coef, label='iou_coef')
plt.plot(val_iou_coef, label='val_iou_coef')
plt.title('model iou coef')
plt.ylabel('iou coef')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./u_net_update_plot/iou_coef.png')
plt.close()

plt.figure()
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./u_net_update_plot/acc.png')
plt.close()
