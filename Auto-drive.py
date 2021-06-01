from utils import *
from sklearn.model_selection import train_test_split
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

#Data Imorting
path = "simulationData"
data = importDataInfo(path)

#Data visualization
data = balanceData(data, display=False)

#Loading data
imagesPath, steering = loadData(path, data)
print(imagesPath[0], steering[0])

#train_val split
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print("Total Train Images: ", len(xTrain))
print("Total Val Images: ", len(xVal))

#model creation
model = createModel()
model.summary()

H = model.fit(batchGen(xTrain, yTrain, 100, True), steps_per_epoch=300, epochs = 10,
          validation_data = batchGen(xVal, yVal, 100, False), validation_steps = 200)

model.save('model.h5')
print("Model saved!!")

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()