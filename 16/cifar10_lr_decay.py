#python cifar10_lr_decay.py --output output/lr_decay_f0.25_plot.png

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


def step_decay(epoch):
	initAlpha=0.01
	factor=0.25
	dropEvery=5
	
	alpha=initAlpha*(factor**np.floor((1+epoch)/dropEvery))
	
	return float(alpha)
	
ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to output loss/accuracy plot")
args=vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX,trainY),(testX,testY))=cifar10.load_data()

trainX=trainX.astype("float")/255.0
testX=testX.astype("float")/255.0

lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

labelNames=["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

callbacks=[LearningRateScheduler(step_decay)]

opt=SGD(lr=0.01,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(width=32,height=32,,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,epochs=40,callbacks=callbacks,verbose=1)

print("[INFO] evaluating network...")
predictions=model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])






















