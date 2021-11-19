from lenet import LeNet
from keras.utils.vis_utils import plot_model

model=LeNet.build(28,28,1,10)
plot_model(model,to_file="lenet.png",show_shapes=True)

#output not of previous now only Sequential is written
