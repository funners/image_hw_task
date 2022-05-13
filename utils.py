# -*- coding = utf-8 -*-
# @File_name = utils
def predict(model, images):
    pred = model.predict(images)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    return pred